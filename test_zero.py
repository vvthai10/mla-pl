import os
import argparse
import random
import numpy as np
import torch
from torch.nn import functional as F
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
from CLIP.learnable_prompt import PromptMaker
from dataset.medical_zero import MedTestDataset, MedTrainDataset
from CLIP.clip import create_model
from CLIP.multi_level_adapter import MultiLevelAdapters
from loss import FocalLoss, BinaryDiceLoss
from utils import encode_text_with_prompt_ensemble
from prompt import REAL_NAME

import warnings

from visualization import visualizer

warnings.filterwarnings("ignore")

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")

CLASS_INDEX = {
    "Bone_v3": 4,
    "Brain": 3,
    "Liver": 2,
    "Retina_RESC": 1,
    "Retina_OCT2017": -1,
    "Chest": -2,
    "Histopathology": -3,
}
CLASS_INDEX_INV = {
    4: "Bone_v3",
    3: "Brain",
    2: "Liver",
    1: "Retina_RESC",
    -1: "Retina_OCT2017",
    -2: "Chest",
    -3: "Histopathology",
}


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    parser = argparse.ArgumentParser(description="Testing")
    parser.add_argument(
        "--model_name",
        type=str,
        default="ViT-L-14-336",
        help="ViT-B-16-plus-240, ViT-L-14-336",
    )
    parser.add_argument(
        "--pretrain", type=str, default="openai", help="laion400m, openai"
    )
    parser.add_argument("--obj", type=str, default="Brain")
    parser.add_argument("--data_path", type=str, default="./data/")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--img_size", type=int, default=240)
    parser.add_argument(
        "--features_list",
        type=int,
        nargs="+",
        default=[6, 12, 18, 24],
        help="features used",
    )
    parser.add_argument("--seed", type=int, default=111)
    parser.add_argument("--ckpt_path", type=str, default="./ckpt/zero-shot/Brain.pth")
    parser.add_argument("--visualize_path", type=str, default=None)

    args = parser.parse_args()

    setup_seed(args.seed)

    # fixed feature extractor
    clip_model = create_model(
        model_name=args.model_name,
        img_size=args.img_size,
        device=device,
        pretrained=args.pretrain,
        require_pretrained=True,
    )
    clip_model.eval()

    model = MultiLevelAdapters(clip_model=clip_model, features=args.features_list).to(
        device
    )
    model.eval()

    clip_model.device = device
    prompt_maker = PromptMaker(
        clip_model=clip_model, n_ctx=8, CSC=True, class_token_position=["end"]
    ).to(device)
    prompt_maker.eval()

    checkpoint = torch.load(args.ckpt_path)
    model.seg_adapters.load_state_dict(checkpoint["state_dict"]["seg_adapters"])
    model.det_adapters.load_state_dict(checkpoint["state_dict"]["det_adapters"])
    prompt_maker.prompt_learner.load_state_dict(
        checkpoint["state_dict"]["prompt_learner"]
    )

    # load dataset and loader
    kwargs = {"num_workers": 4, "pin_memory": True} if use_cuda else {}
    test_dataset = MedTestDataset(args.data_path, args.obj, args.img_size)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=1, shuffle=False, **kwargs
    )

    test(args, model, test_loader, prompt_maker)


def test(args, seg_model, test_loader, prompt_maker):
    gt_list = []
    gt_mask_list = []
    image_scores = []
    segment_scores = []

    for image, y, mask, pathes in tqdm(test_loader, position=0, leave=True):
        image = image.to(device)
        mask[mask > 0.5], mask[mask <= 0.5] = 1, 0

        with torch.no_grad(), torch.cuda.amp.autocast():
            ori_seg_patch_tokens, ori_det_patch_tokens = seg_model(image)
            ori_seg_patch_tokens = [p[0, 1:, :] for p in ori_seg_patch_tokens]
            ori_det_patch_tokens = [p[0, 1:, :] for p in ori_det_patch_tokens]

            prompts_feat = prompt_maker()

            # image
            anomaly_score = 0
            patch_tokens = ori_det_patch_tokens.copy()
            for layer in range(len(patch_tokens)):
                patch_tokens[layer] /= patch_tokens[layer].norm(dim=-1, keepdim=True)
                anomaly_map = (100.0 * patch_tokens[layer] @ prompts_feat).unsqueeze(0)
                anomaly_map = torch.softmax(anomaly_map, dim=-1)[:, :, 1]
                anomaly_score += anomaly_map.mean()
            image_scores.append(anomaly_score.cpu())

            # pixel
            patch_tokens = ori_seg_patch_tokens
            anomaly_maps = []
            for layer in range(len(patch_tokens)):
                patch_tokens[layer] /= patch_tokens[layer].norm(dim=-1, keepdim=True)
                anomaly_map = (100.0 * patch_tokens[layer] @ prompts_feat).unsqueeze(0)
                B, L, C = anomaly_map.shape
                H = int(np.sqrt(L))
                anomaly_map = F.interpolate(
                    anomaly_map.permute(0, 2, 1).view(B, 2, H, H),
                    size=args.img_size,
                    mode="bilinear",
                    align_corners=True,
                )
                anomaly_map = torch.softmax(anomaly_map, dim=1)[:, 1, :, :]
                anomaly_maps.append(anomaly_map.cpu().numpy())
            final_score_map = np.sum(anomaly_maps, axis=0)

            if CLASS_INDEX[args.obj] > 0:
                visualizer(
                    pathes,
                    final_score_map,
                    args.visualize_path,
                )

            gt_mask_list.append(mask.squeeze().cpu().detach().numpy())
            gt_list.extend(y.cpu().detach().numpy())
            segment_scores.append(final_score_map)

    gt_list = np.array(gt_list)
    gt_mask_list = np.asarray(gt_mask_list)
    gt_mask_list = (gt_mask_list > 0).astype(np.int_)

    segment_scores = np.array(segment_scores)
    image_scores = np.array(image_scores)

    segment_scores = (segment_scores - segment_scores.min()) / (
        segment_scores.max() - segment_scores.min()
    )
    image_scores = (image_scores - image_scores.min()) / (
        image_scores.max() - image_scores.min()
    )

    img_roc_auc_det = roc_auc_score(gt_list, image_scores)
    print(f"{args.obj} AUC : {round(img_roc_auc_det,4)}")

    if CLASS_INDEX[args.obj] > 0:
        seg_roc_auc = roc_auc_score(gt_mask_list.flatten(), segment_scores.flatten())
        print(f"{args.obj} pAUC : {round(seg_roc_auc,4)}")
        return seg_roc_auc + img_roc_auc_det
    else:
        return img_roc_auc_det


if __name__ == "__main__":
    main()

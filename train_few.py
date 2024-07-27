import os
import argparse
import random
import numpy as np
import torch
from torch.nn import functional as F
from tqdm import tqdm
from dataset.medical_few import MedDataset
from CLIP import create_model, PromptMaker, MultiLevelAdapters
from sklearn.metrics import roc_auc_score
from loss import FocalLoss, BinaryDiceLoss
from utils import augment, cos_sim
from pathlib import Path

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import warnings

warnings.filterwarnings("ignore")

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
print(torch.__version__)

CLASS_INDEX = {
    "Bone": 4,
    "Brain": 3,
    "Liver": 2,
    "Retina_RESC": 1,
    "Retina_OCT2017": -1,
    "Chest": -2,
    "Histopathology": -3,
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
    parser.add_argument("--obj", type=str, default="Liver")
    parser.add_argument("--data_path", type=str, default="./data/")
    parser.add_argument("--ckpt_path", type=str, default="./ckpt/few-shot/")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--img_size", type=int, default=240)
    parser.add_argument("--epoch", type=int, default=50, help="epochs")
    parser.add_argument(
        "--learning_rate", type=float, default=1e-4, help="learning rate"
    )
    parser.add_argument(
        "--features_list",
        type=int,
        nargs="+",
        default=[6, 12, 18, 24],
        help="features used",
    )
    parser.add_argument("--seed", type=int, default=111)
    parser.add_argument("--continue_path", type=str, default=None, help="continue")
    parser.add_argument("--shot", type=int, default=4)
    parser.add_argument("--iterate", type=int, default=0)
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
    prompt_maker.train()

    # map_maker = MapMaker(image_size=args.config.image_size).to(device)
    continue_epoch = 0
    best_result = 0

    if args.continue_path:
        checkpoint = torch.load(args.continue_path)
        model.seg_adapters.load_state_dict(checkpoint["state_dict"]["seg_adapters"])
        model.det_adapters.load_state_dict(checkpoint["state_dict"]["det_adapters"])
        prompt_maker.prompt_learner.load_state_dict(
            checkpoint["state_dict"]["prompt_learner"]
        )

        continue_epoch = checkpoint["epoch"] + 1
        
        print("Best result: ")
        if checkpoint["best"]:
            best_result = checkpoint["best"]
        else:
            best_result = checkpoint["AUC"]
            print("AUC: ", best_result)
            if checkpoint["pAUC"]:
                best_result += checkpoint["pAUC"]
                print("pAUC: ", checkpoint["pAUC"])

    for name, param in model.named_parameters():
        param.requires_grad = True

    # optimizer for only adapters
    seg_optimizer = torch.optim.Adam(
        list(model.seg_adapters.parameters()), lr=args.learning_rate, betas=(0.5, 0.999)
    )
    det_optimizer = torch.optim.Adam(
        list(model.det_adapters.parameters()), lr=args.learning_rate, betas=(0.5, 0.999)
    )

    learnable_prompt_optimizer = torch.optim.Adam(
        [
            {"params": prompt_maker.prompt_learner.parameters(), "lr": 0.001},
        ],
        lr=0.001,
        betas=(0.5, 0.999),
    )

    # load test dataset
    kwargs = {"num_workers": 64, "pin_memory": True} if use_cuda else {}
    test_dataset = MedDataset(
        args.data_path, args.obj, args.img_size, args.shot, args.iterate
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=1, shuffle=False, **kwargs
    )

    # few-shot image augmentation
    augment_abnorm_img, augment_abnorm_mask = augment(
        test_dataset.fewshot_abnorm_img, test_dataset.fewshot_abnorm_mask
    )
    augment_normal_img, augment_normal_mask = augment(test_dataset.fewshot_norm_img)

    augment_fewshot_img = torch.cat([augment_abnorm_img, augment_normal_img], dim=0)
    augment_fewshot_mask = torch.cat([augment_abnorm_mask, augment_normal_mask], dim=0)

    augment_fewshot_label = torch.cat(
        [
            torch.Tensor([1] * len(augment_abnorm_img)),
            torch.Tensor([0] * len(augment_normal_img)),
        ],
        dim=0,
    )

    train_dataset = torch.utils.data.TensorDataset(
        augment_fewshot_img, augment_fewshot_mask, augment_fewshot_label
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=1, shuffle=True, **kwargs
    )

    # memory bank construction
    support_dataset = torch.utils.data.TensorDataset(augment_normal_img)
    support_loader = torch.utils.data.DataLoader(
        support_dataset, batch_size=1, shuffle=True, **kwargs
    )

    # losses
    loss_focal = FocalLoss()
    loss_dice = BinaryDiceLoss()
    loss_bce = torch.nn.BCEWithLogitsLoss()

    for epoch in range(continue_epoch, args.epoch):
        print("epoch ", epoch, ":")

        loss_list = []
        for image, gt, label in train_loader:
            image = image.to(device)
            with torch.cuda.amp.autocast():
                seg_patch_tokens, det_patch_tokens = model(image)

                seg_patch_tokens = [p[0, 1:, :] for p in seg_patch_tokens]
                det_patch_tokens = [p[0, 1:, :] for p in det_patch_tokens]

                prompts_feat = prompt_maker()

                # det loss
                det_loss = 0
                image_label = label.to(device)

                # Image level
                for layer in range(len(det_patch_tokens)):
                    det_patch_tokens[layer] = det_patch_tokens[
                        layer
                    ] / det_patch_tokens[layer].norm(dim=-1, keepdim=True)
                    anomaly_map = (
                        100.0 * det_patch_tokens[layer] @ prompts_feat
                    ).unsqueeze(0)
                    anomaly_map = torch.softmax(anomaly_map, dim=-1)[:, :, 1]
                    anomaly_score = torch.mean(anomaly_map, dim=-1)
                    det_loss += loss_bce(anomaly_score, image_label)

                # Pixel level
                if CLASS_INDEX[args.obj] > 0:
                    seg_loss = 0
                    mask = gt.squeeze(0).to(device)
                    mask[mask > 0.5], mask[mask <= 0.5] = 1, 0
                    for layer in range(len(seg_patch_tokens)):
                        seg_patch_tokens[layer] = seg_patch_tokens[
                            layer
                        ] / seg_patch_tokens[layer].norm(dim=-1, keepdim=True)
                        anomaly_map = (
                            100.0 * seg_patch_tokens[layer] @ prompts_feat
                        ).unsqueeze(0)
                        B, L, C = anomaly_map.shape
                        H = int(np.sqrt(L))
                        anomaly_map = F.interpolate(
                            anomaly_map.permute(0, 2, 1).view(B, 2, H, H),
                            size=args.img_size,
                            mode="bilinear",
                            align_corners=True,
                        )
                        anomaly_map = torch.softmax(anomaly_map, dim=1)
                        seg_loss += loss_focal(anomaly_map, mask)
                        seg_loss += loss_dice(anomaly_map[:, 1, :, :], mask)

                    loss = seg_loss + det_loss
                    loss.requires_grad_(True)
                    seg_optimizer.zero_grad()
                    det_optimizer.zero_grad()
                    learnable_prompt_optimizer.zero_grad()
                    loss.backward()
                    seg_optimizer.step()
                    det_optimizer.step()
                    learnable_prompt_optimizer.step()

                else:
                    loss = det_loss
                    loss.requires_grad_(True)
                    det_optimizer.zero_grad()
                    learnable_prompt_optimizer.zero_grad()
                    loss.backward()
                    det_optimizer.step()
                    learnable_prompt_optimizer.step()

                loss_list.append(loss.item())

        print("Loss: ", np.mean(loss_list))

        seg_features = []
        det_features = []
        for image in support_loader:
            image = image[0].to(device)
            with torch.no_grad():
                seg_patch_tokens, det_patch_tokens = model(image)
                seg_patch_tokens = [p[0].contiguous() for p in seg_patch_tokens]
                det_patch_tokens = [p[0].contiguous() for p in det_patch_tokens]
                seg_features.append(seg_patch_tokens)
                det_features.append(det_patch_tokens)
        seg_mem_features = [
            torch.cat([seg_features[j][i] for j in range(len(seg_features))], dim=0)
            for i in range(len(seg_features[0]))
        ]
        det_mem_features = [
            torch.cat([det_features[j][i] for j in range(len(det_features))], dim=0)
            for i in range(len(det_features[0]))
        ]

        result = test(
            args,
            model,
            test_loader,
            prompt_maker,
            seg_mem_features,
            det_mem_features,
        )

        ckp_path = os.path.join(args.ckpt_path, f"{args.obj}_lastest.pth")
        os.makedirs(Path(ckp_path).parent, exist_ok=True)
        torch.save(
            {
                "state_dict": {
                    "seg_adapters": model.seg_adapters.state_dict(),
                    "det_adapters": model.det_adapters.state_dict(),
                    "prompt_learner": prompt_maker.prompt_learner.state_dict(),
                },
                "AUC": result[1],
                "pAUC": result[2],
                "best": best_result,
                "epoch": epoch,
            },
            ckp_path,
        )

        if result[0] > best_result:
            best_result = result[0]
            print("Best result\n")
            ckp_path = os.path.join(args.save_path, f"{args.obj}.pth")
            os.makedirs(Path(ckp_path).parent, exist_ok=True)
            torch.save(
                {
                    "state_dict": {
                        "seg_adapters": model.seg_adapters.state_dict(),
                        "det_adapters": model.det_adapters.state_dict(),
                        "prompt_learner": prompt_maker.prompt_learner.state_dict(),
                    },
                    "AUC": result[1],
                    "pAUC": result[2],
                    "best": best_result,
                    "epoch": epoch,
                },
                ckp_path,
            )


def test(
    args,
    model,
    test_loader,
    prompt_maker,
    seg_mem_features,
    det_mem_features,
):
    gt_list = []
    gt_mask_list = []

    det_image_scores_zero = []
    det_image_scores_few = []

    seg_score_map_zero = []
    seg_score_map_few = []

    for image, y, mask, path in tqdm(test_loader, position=0, leave=True):
        image = image.to(device)
        mask[mask > 0.5], mask[mask <= 0.5] = 1, 0

        with torch.no_grad(), torch.cuda.amp.autocast():
            _, seg_patch_tokens, det_patch_tokens = model(image)
            seg_patch_tokens = [p[0, 1:, :] for p in seg_patch_tokens]
            det_patch_tokens = [p[0, 1:, :] for p in det_patch_tokens]

            prompts_feat = prompt_maker(det_patch_tokens)
            if CLASS_INDEX[args.obj] > 0:

                # few-shot, seg head
                anomaly_maps_few_shot = []
                for idx, p in enumerate(seg_patch_tokens):
                    cos = cos_sim(seg_mem_features[idx], p)
                    height = int(np.sqrt(cos.shape[1]))
                    anomaly_map_few_shot = torch.min((1 - cos), 0)[0].reshape(
                        1, 1, height, height
                    )
                    anomaly_map_few_shot = F.interpolate(
                        torch.tensor(anomaly_map_few_shot),
                        size=args.img_size,
                        mode="bilinear",
                        align_corners=True,
                    )
                    anomaly_maps_few_shot.append(anomaly_map_few_shot[0].cpu().numpy())
                score_map_few = np.sum(anomaly_maps_few_shot, axis=0)
                seg_score_map_few.append(score_map_few)

                # zero-shot, seg head
                anomaly_maps = []
                for layer in range(len(seg_patch_tokens)):
                    seg_patch_tokens[layer] /= seg_patch_tokens[layer].norm(
                        dim=-1, keepdim=True
                    )
                    anomaly_map = (
                        100.0 * seg_patch_tokens[layer] @ prompts_feat
                    ).unsqueeze(0)
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
                score_map_zero = np.sum(anomaly_maps, axis=0)
                seg_score_map_zero.append(score_map_zero)

            else:
                # few-shot, det head
                anomaly_maps_few_shot = []
                for idx, p in enumerate(det_patch_tokens):
                    cos = cos_sim(det_mem_features[idx], p)
                    height = int(np.sqrt(cos.shape[1]))
                    anomaly_map_few_shot = torch.min((1 - cos), 0)[0].reshape(
                        1, 1, height, height
                    )
                    anomaly_map_few_shot = F.interpolate(
                        torch.tensor(anomaly_map_few_shot),
                        size=args.img_size,
                        mode="bilinear",
                        align_corners=True,
                    )
                    anomaly_maps_few_shot.append(anomaly_map_few_shot[0].cpu().numpy())
                anomaly_map_few_shot = np.sum(anomaly_maps_few_shot, axis=0)
                score_few_det = anomaly_map_few_shot.mean()
                det_image_scores_few.append(score_few_det)

                # zero-shot, det head
                anomaly_score = 0
                for layer in range(len(det_patch_tokens)):
                    det_patch_tokens[layer] /= det_patch_tokens[layer].norm(
                        dim=-1, keepdim=True
                    )
                    anomaly_map = (
                        100.0 * det_patch_tokens[layer] @ prompts_feat
                    ).unsqueeze(0)
                    anomaly_map = torch.softmax(anomaly_map, dim=-1)[:, :, 1]
                    anomaly_score += anomaly_map.mean()
                det_image_scores_zero.append(anomaly_score.cpu().numpy())

            gt_mask_list.append(mask.squeeze().cpu().detach().numpy())
            gt_list.extend(y.cpu().detach().numpy())

    gt_list = np.array(gt_list)
    gt_mask_list = np.asarray(gt_mask_list)
    gt_mask_list = (gt_mask_list > 0).astype(np.int_)

    if CLASS_INDEX[args.obj] > 0:

        seg_score_map_zero = np.array(seg_score_map_zero)
        seg_score_map_few = np.array(seg_score_map_few)

        seg_score_map_zero = (seg_score_map_zero - seg_score_map_zero.min()) / (
            seg_score_map_zero.max() - seg_score_map_zero.min()
        )
        seg_score_map_few = (seg_score_map_few - seg_score_map_few.min()) / (
            seg_score_map_few.max() - seg_score_map_few.min()
        )

        segment_scores = 0.5 * seg_score_map_zero + 0.5 * seg_score_map_few
        seg_roc_auc = roc_auc_score(gt_mask_list.flatten(), segment_scores.flatten())
        print(f"{args.obj} pAUC : {round(seg_roc_auc,4)}")

        segment_scores_flatten = segment_scores.reshape(segment_scores.shape[0], -1)
        roc_auc_im = roc_auc_score(gt_list, np.max(segment_scores_flatten, axis=1))
        print(f"{args.obj} AUC : {round(roc_auc_im, 4)}")

        return roc_auc_im + seg_roc_auc, roc_auc_im, seg_roc_auc

    else:

        det_image_scores_zero = np.array(det_image_scores_zero)
        det_image_scores_few = np.array(det_image_scores_few)

        det_image_scores_zero = (
            det_image_scores_zero - det_image_scores_zero.min()
        ) / (det_image_scores_zero.max() - det_image_scores_zero.min())
        det_image_scores_few = (det_image_scores_few - det_image_scores_few.min()) / (
            det_image_scores_few.max() - det_image_scores_few.min()
        )

        image_scores = 0.5 * det_image_scores_zero + 0.5 * det_image_scores_few
        img_roc_auc_det = roc_auc_score(gt_list, image_scores)
        print(f"{args.obj} AUC : {round(img_roc_auc_det,4)}")

        return img_roc_auc_det, img_roc_auc_det, None


if __name__ == "__main__":
    main()

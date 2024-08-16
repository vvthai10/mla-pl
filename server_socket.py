import argparse
import base64
import io
import math
import random

import numpy as np
import torch
import torch.utils.data
from flask import Flask, jsonify, request
from PIL import Image
from sklearn.metrics import roc_auc_score
from torch.nn import functional as F
from torchvision import transforms
from tqdm import tqdm

from CLIP.clip import create_model
from CLIP.learnable_prompt import PromptMaker
from CLIP.multi_level_adapter import MultiLevelAdapters
from dataset.medical_few import CLASS_INDEX, MedDataset
from dataset.medical_zero import MedTestDataset, MedTrainDataset
from server_util import visualizer
from utils import augment, cos_sim

app = Flask(__name__)
from flask_cors import CORS

CORS(app)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Load the model
obj = "Bone_v3"
shot = "16-shot"
use_cuda = torch.cuda.is_available()
kwargs = {"num_workers": 4, "pin_memory": True} if use_cuda else {}
if use_cuda:
    print("use cuda")

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def setup_model():
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

    args = parser.parse_args()
    setup_seed(args.seed)

    preprocess = transforms.Compose(
        [
            transforms.Resize((args.img_size, args.img_size), Image.BICUBIC),
            transforms.ToTensor(),
        ]
    )

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
    nshot = shot.replace("-shot", "")
    Obj = obj.capitalize()

    ckpt_path = f"./ckpt/few-shot/{nshot}/{Obj}.pth"
    data_path = "./data"
    checkpoint = torch.load(ckpt_path)
    model.seg_adapters.load_state_dict(checkpoint["state_dict"]["seg_adapters"])
    model.det_adapters.load_state_dict(checkpoint["state_dict"]["det_adapters"])
    prompt_maker.prompt_learner.load_state_dict(
        checkpoint["state_dict"]["prompt_learner"]
    )

    test_dataset = MedDataset(data_path, Obj, 240, int(nshot), -1)

    # few-shot image augmentation
    augment_normal_img, augment_normal_mask = augment(test_dataset.fewshot_norm_img)

    # memory bank construction
    support_dataset = torch.utils.data.TensorDataset(augment_normal_img)
    support_loader = torch.utils.data.DataLoader(
        support_dataset, batch_size=1, shuffle=True, **kwargs
    )

    seg_features = []
    det_features = []
    # for image in tqdm(support_loader, desc="Load support dataset"):
    for image in tqdm(support_loader, desc="Load support dataset"):
        image = image[0].to(device)
        with torch.no_grad():
            seg_patch_tokens, det_patch_tokens = model(image)
            seg_patch_tokens = [p[0].contiguous() for p in seg_patch_tokens]
            det_patch_tokens = [p[0].contiguous() for p in det_patch_tokens]
            seg_features.append(seg_patch_tokens)
            det_features.append(det_patch_tokens)

        print("1")

    seg_mem_features = [
        torch.cat([seg_features[j][i] for j in range(len(seg_features))], dim=0)
        for i in range(len(seg_features[0]))
    ]
    det_mem_features = [
        torch.cat([det_features[j][i] for j in range(len(det_features))], dim=0)
        for i in range(len(det_features[0]))
    ]
    return model, prompt_maker, preprocess, args, seg_mem_features, det_mem_features


model, prompt_maker, preprocess, args, seg_mem_features, det_mem_features = (
    None,
    None,
    None,
    None,
    None,
    None,
)


@app.route("/anomaly_detection", methods=["POST"])
def anomaly_detection():
    # obj = request.form.get("obj")
    # shot = request.form.get("shot")

    nshot = shot.replace("-shot", "")
    Obj = obj.capitalize()

    image_data = request.files["image"]
    image = Image.open(image_data.stream).convert("RGB")
    print(image.size)

    size_ori_image = image.size
    # Tiền xử lý ảnh
    image_np = np.array(image)
    image = preprocess(image).unsqueeze(0)
    image = image.to(device)

    prompts_feat = prompt_maker()

    det_image_scores_zero = []
    det_image_scores_few = []

    seg_score_map_zero = []
    seg_score_map_few = []

    with torch.no_grad(), torch.cuda.amp.autocast():
        seg_patch_tokens, det_patch_tokens = model(image)
        seg_patch_tokens = [p[0, 1:, :] for p in seg_patch_tokens]
        det_patch_tokens = [p[0, 1:, :] for p in det_patch_tokens]
        prompts_feat = prompt_maker()

        if CLASS_INDEX[Obj] <= 0:
            return jsonify({"error": "Error obj!"}), 200

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
            anomaly_map = (100.0 * seg_patch_tokens[layer] @ prompts_feat).unsqueeze(0)
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

        seg_score_map_zero = np.array(seg_score_map_zero)
        seg_score_map_few = np.array(seg_score_map_few)
        seg_score_map_zero = (seg_score_map_zero - seg_score_map_zero.min()) / (
            seg_score_map_zero.max() - seg_score_map_zero.min()
        )
        seg_score_map_few = (seg_score_map_few - seg_score_map_few.min()) / (
            seg_score_map_few.max() - seg_score_map_few.min()
        )

        segment_scores = 0.5 * seg_score_map_zero + 0.5 * seg_score_map_few

        segment_scores_flatten = segment_scores.reshape(segment_scores.shape[0], -1)

        # print(segment_scores_flatten)
        if np.mean(segment_scores_flatten) <= 0.1:
            return (
                jsonify(
                    {"error": "This patient's photograph contains no abnormalities."}
                ),
                200,
            )

        image_result = visualizer(size_ori_image, image_np, segment_scores[0])

        # print("shape: ", segment_scores_flatten.shape)

        # print("score mean:", np.mean(segment_scores_flatten))
        # print("score:", np.max(segment_scores_flatten))
        return jsonify(image_result), 200

@app.route("/test", methods=["GET"])
def test():
    return jsonify({"message": "Server is running"}), 200

if __name__ == "__main__":
    model, prompt_maker, preprocess, args, seg_mem_features, det_mem_features = setup_model()
    app.run(host="localhost", port=5000)

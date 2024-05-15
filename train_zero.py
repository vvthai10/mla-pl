import os
import argparse
import random
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
from scipy.ndimage import gaussian_filter
from dataset.medical_zero import MedTestDataset, MedTrainDataset
from CLIP.clip import create_model
from CLIP.tokenizer import tokenize
from CLIP.adapter import CLIP_Inplanted
from PIL import Image
from sklearn.metrics import precision_recall_curve
from loss import FocalLoss, BinaryDiceLoss
from utils import augment, encode_text_with_prompt_ensemble
from prompt import REAL_NAME
import matplotlib.pyplot as plt

import warnings

warnings.filterwarnings("ignore")

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")

CLASS_INDEX = {'Brain': 3, 'Liver': 2, 'Retina_RESC': 1}  # , 'Retina_OCT2017':-1, 'Chest':-2, 'Histopathology':-3
CLASS_INDEX_INV = {3: 'Brain', 2: 'Liver', 1: 'Retina_RESC'}  # , -1:'Retina_OCT2017', -2:'Chest', -3:'Histopathology'


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    parser = argparse.ArgumentParser(description='Testing')
    parser.add_argument('--model_name', type=str, default='ViT-L-14-336', help="ViT-B-16-plus-240, ViT-L-14-336")
    parser.add_argument('--pretrain', type=str, default='openai', help="laion400m, openai")
    parser.add_argument('--obj', type=str, default='Retina_RESC')
    parser.add_argument('--data_path', type=str, default='./data/')
    parser.add_argument('--ckpt_path', type=str, default='./ckpt/')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--img_size', type=int, default=240)
    parser.add_argument("--epoch", type=int, default=50, help="epochs")
    parser.add_argument("--learning_rate", type=float, default=0.0001, help="learning rate")
    parser.add_argument("--features_list", type=int, nargs="+", default=[6, 12, 18, 24], help="features used")
    parser.add_argument('--seed', type=int, default=111)
    args = parser.parse_args()

    setup_seed(args.seed)

    # fixed feature extractor
    clip_model = create_model(model_name=args.model_name, img_size=args.img_size, device=device,
                              pretrained=args.pretrain, require_pretrained=True)
    clip_model.eval()

    model = CLIP_Inplanted(clip_model=clip_model, features=args.features_list).to(device)
    model.eval()

    for name, param in model.named_parameters():
        param.requires_grad = True

    # optimizer for only adapters
    seg_optimizer = torch.optim.Adam(list(model.seg_adapters.parameters()), lr=args.learning_rate, betas=(0.5, 0.999))
    # det_optimizer = torch.optim.Adam(list(model.det_adapters.parameters()), lr=args.learning_rate, betas=(0.5, 0.999))
    second_seg_optimizer = torch.optim.Adam(list(model.second_seg_adapter.parameters()), lr=args.learning_rate,
                                            betas=(0.5, 0.999))
    det_optimizer = torch.optim.Adam(list(model.det_adapter.parameters()), lr=args.learning_rate, betas=(0.5, 0.999))

    # load dataset and loader
    kwargs = {'num_workers': 4, 'pin_memory': True} if use_cuda else {}
    train_dataset = MedTrainDataset(args.data_path, args.obj, args.img_size, args.batch_size)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=True, **kwargs)

    test_dataset = MedTestDataset(args.data_path, args.obj, args.img_size)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, **kwargs)

    # losses
    loss_focal = FocalLoss()
    loss_dice = BinaryDiceLoss()
    loss_bce = torch.nn.BCEWithLogitsLoss()

    text_feature_list = [0]
    # text prompt
    with torch.cuda.amp.autocast(), torch.no_grad():
        for i in [1, 2, 3]:  # ,-3,-2,-1
            text_feature = encode_text_with_prompt_ensemble(clip_model, REAL_NAME[CLASS_INDEX_INV[i]], device)
            text_feature_list.append(text_feature)

    save_score = 0.0
    torch.autograd.set_detect_anomaly(True)
    for epoch in range(args.epoch):
        print('epoch', epoch, ':')

        loss_list = []
        idx = 0
        for (image, image_label, mask, seg_idx) in tqdm(train_loader):
            if idx % 10 == 0: #idx % (len(train_loader) // 5) == 0
                print("\nTest...")
                score = test(args, model, test_dataset, test_loader, text_feature_list[CLASS_INDEX[args.obj]])
                if score >= save_score:
                    save_score = score
                    ckp_path = f'{args.ckpt_path}/zero-shot/{args.obj}.pth'
                    torch.save({'seg_adapters': model.seg_adapters.state_dict(),
                                'det_adapter': model.det_adapter.state_dict(),
                                'second_seg_adapter': model.second_seg_adapter.state_dict()},
                                ckp_path)
                    print(f'best epoch found: epoch {epoch} batch {idx}')
                print('score test: ' + str(score) + '\n')
            idx += 1

            image = image.squeeze(0).to(device)

            mask = mask.squeeze(0).to(device)
            mask[mask > 0.5], mask[mask <= 0.5] = 1, 0

            with torch.cuda.amp.autocast():
                _, seg_patch_tokens = model(image)  # , det_patch_tokens
                seg_patch_tokens = [p[0, 1:, :] for p in seg_patch_tokens]
                # det_patch_tokens = [p[0, 1:, :] for p in det_patch_tokens]

                # pixel level
                seg_loss = 0
                anomaly_maps = []
                for layer in range(len(seg_patch_tokens)):
                    seg_patch_tokens[layer] = seg_patch_tokens[layer] / seg_patch_tokens[layer].norm(dim=-1,
                                                                                                     keepdim=True)
                    # print(seg_patch_tokens[layer].shape, text_feature_list[seg_idx].shape) # torch.Size([289, 768]) torch.Size([768, 2])
                    anomaly_map = (100.0 * seg_patch_tokens[layer] @ text_feature_list[seg_idx]).unsqueeze(0)
                    B, L, C = anomaly_map.shape
                    H = int(np.sqrt(L))
                    anomaly_map = F.interpolate(anomaly_map.permute(0, 2, 1).view(B, 2, H, H),
                                                size=args.img_size, mode='bilinear', align_corners=True)
                    anomaly_map = torch.softmax(anomaly_map, dim=1)
                    anomaly_maps.append(anomaly_map[:, 1, :, :])
                    seg_loss += loss_focal(anomaly_map, mask)
                    seg_loss += loss_dice(anomaly_map[:, 1, :, :], mask)

                # TODO: Calculation second adapter, batch = 1
                batch = image.shape[0]
                second_input = []
                for b in range(batch):
                    b_img = image[b].clone()
                    ori_img = b_img * 255
                    ori_img = ori_img.cpu().numpy().astype(np.uint8)
                    ori_img = np.transpose(ori_img, (1, 2, 0))

                    anomaly_map_combined = anomaly_maps[0]
                    for anomaly_map in anomaly_maps[1:]:
                        anomaly_map_combined = anomaly_map_combined + anomaly_map.clone()
                    anomaly_map_combined = torch.clamp(anomaly_map_combined, min=0, max=1)

                    anomaly_map_combined = anomaly_map_combined.squeeze().cpu().detach().numpy()
                    rows_to_keep = np.any(anomaly_map_combined > 0.5, axis=1)
                    cols_to_keep = np.any(anomaly_map_combined > 0.5, axis=0)

                    mask_detected = np.zeros_like(ori_img, dtype=bool)

                    start_row, end_row = np.where(rows_to_keep)[0][[0, -1]]
                    start_col, end_col = np.where(cols_to_keep)[0][[0, -1]]
                    mask_detected[start_row:end_row + 1, start_col:end_col + 1, :] = True

                    local_image = np.where(mask_detected, ori_img, 0)
                    local_image = Image.fromarray(local_image)
                    local_image = train_dataset.transform_x(local_image)
                    second_input += [local_image]
                second_input = torch.stack(second_input)
                second_input = second_input.to(device)

                # TODO: Cal Loss
                seg_adapt_med, det_adapt_med = model.forward_second_adapter(second_input)
                seg_adapt_med = seg_adapt_med[0, 1:, :].clone()
                det_adapt_med = det_adapt_med[0, 1:, :].clone()

                det_loss = 0
                image_label = image_label.squeeze(0).to(device)
                det_adapt_med = det_adapt_med / det_adapt_med.norm(dim=-1, keepdim=True)
                anomaly_map_det = (100.0 * det_adapt_med @ text_feature_list[seg_idx]).unsqueeze(0)
                anomaly_map_det = torch.softmax(anomaly_map_det, dim=-1)[:, :, 1]
                anomaly_score = torch.mean(anomaly_map_det, dim=-1)
                det_loss = loss_bce(anomaly_score, image_label)

                seg_second_loss = 0
                seg_adapt_med = seg_adapt_med / seg_adapt_med.norm(dim=-1, keepdim=True)
                # print(seg_patch_tokens[layer].shape, text_feature_list[seg_idx].shape) # torch.Size([289, 768]) torch.Size([768, 2])
                anomaly_map_seg_second = (100.0 * seg_adapt_med @ text_feature_list[seg_idx]).unsqueeze(0)
                B, L, C = anomaly_map_seg_second.shape
                H = int(np.sqrt(L))
                anomaly_map_seg_second = F.interpolate(anomaly_map_seg_second.permute(0, 2, 1).view(B, 2, H, H),
                                            size=args.img_size, mode='bilinear', align_corners=True)
                anomaly_map_seg_second = torch.softmax(anomaly_map_seg_second, dim=1)
                seg_second_loss += loss_focal(anomaly_map_seg_second, mask)
                seg_second_loss += loss_dice(anomaly_map_seg_second[:, 1, :, :], mask)

                loss = seg_loss  # = focal(seg_out, mask) + bce(det_out, y)
                loss.requires_grad_(True)
                seg_optimizer.zero_grad()
                second_seg_optimizer.zero_grad()
                det_optimizer.zero_grad()
                loss.backward()
                seg_optimizer.step()
                second_seg_optimizer.step()
                det_optimizer.step()

                loss_list.append(loss.item())

        train_dataset.shuffle_dataset()
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=True, **kwargs)

        # logs
        print("Loss: ", np.mean(loss_list))


def test(args, seg_model, test_dataset, test_loader, text_features):
    gt_list = []
    gt_mask_list = []
    image_scores = []
    segment_scores = []

    for (image, y, mask) in tqdm(test_loader):
        image = image.to(device)
        mask[mask > 0.5], mask[mask <= 0.5] = 1, 0

        with torch.no_grad(), torch.cuda.amp.autocast():
            _, ori_seg_patch_tokens = seg_model(image) #, ori_det_patch_tokens
            ori_seg_patch_tokens = [p[0, 1:, :] for p in ori_seg_patch_tokens]
            # ori_det_patch_tokens = [p[0, 1:, :] for p in ori_det_patch_tokens]

            # pixel
            patch_tokens = ori_seg_patch_tokens
            anomaly_maps = []
            for layer in range(len(patch_tokens)):
                patch_tokens[layer] /= patch_tokens[layer].norm(dim=-1, keepdim=True)
                anomaly_map = (100.0 * patch_tokens[layer] @ text_features).unsqueeze(0)
                B, L, C = anomaly_map.shape
                H = int(np.sqrt(L))
                anomaly_map = F.interpolate(anomaly_map.permute(0, 2, 1).view(B, 2, H, H),
                                            size=args.img_size, mode='bilinear', align_corners=True)
                anomaly_map = torch.softmax(anomaly_map, dim=1)[:, 1, :, :]
                anomaly_maps.append(anomaly_map)

            batch = image.shape[0]
            second_input = []
            for b in range(batch):
                b_img = image[b].clone()
                ori_img = b_img * 255
                ori_img = ori_img.cpu().numpy().astype(np.uint8)
                ori_img = np.transpose(ori_img, (1, 2, 0))

                anomaly_map_combined = anomaly_maps[0]
                for anomaly_map in anomaly_maps[1:]:
                    anomaly_map_combined = anomaly_map_combined + anomaly_map
                anomaly_map_combined = torch.clamp(anomaly_map_combined, min=0, max=1)

                anomaly_map_combined = anomaly_map_combined.squeeze().cpu().detach().numpy()
                rows_to_keep = np.any(anomaly_map_combined > 0.5, axis=1)
                cols_to_keep = np.any(anomaly_map_combined > 0.5, axis=0)

                mask_detected = np.zeros_like(ori_img, dtype=bool)

                start_row, end_row = np.where(rows_to_keep)[0][[0, -1]]
                start_col, end_col = np.where(cols_to_keep)[0][[0, -1]]
                mask_detected[start_row:end_row + 1, start_col:end_col + 1, :] = True

                local_image = np.where(mask_detected, ori_img, 0)
                local_image = Image.fromarray(local_image)
                local_image = test_dataset.transform_x(local_image)
                second_input += [local_image]
            second_input = torch.stack(second_input)
            second_input = second_input.to(device)

            seg_adapt_med, det_adapt_med = seg_model.forward_second_adapter(second_input)
            seg_adapt_med = seg_adapt_med[0, 1:, :].clone()
            det_adapt_med = det_adapt_med[0, 1:, :].clone()

            # image
            anomaly_score = 0
            patch_token = det_adapt_med.clone()
            patch_token /= patch_token.norm(dim=-1, keepdim=True)
            anomaly_map = (100.0 * patch_token @ text_features).unsqueeze(0)
            anomaly_map = torch.softmax(anomaly_map, dim=-1)[:, :, 1]
            anomaly_score += anomaly_map.mean()
            image_scores.append(anomaly_score.cpu())

            patch_token = seg_adapt_med.clone()
            patch_token /= patch_token.norm(dim=-1, keepdim=True)
            anomaly_map = (100.0 * patch_token @ text_features).unsqueeze(0)
            B, L, C = anomaly_map.shape
            H = int(np.sqrt(L))
            anomaly_map = F.interpolate(anomaly_map.permute(0, 2, 1).view(B, 2, H, H),
                                        size=args.img_size, mode='bilinear', align_corners=True)
            anomaly_map = torch.softmax(anomaly_map, dim=1)[:, 1, :, :]
            final_score_map = anomaly_map.cpu().numpy()

            gt_mask_list.append(mask.squeeze().cpu().detach().numpy())
            gt_list.extend(y.cpu().detach().numpy())
            segment_scores.append(final_score_map)

    gt_list = np.array(gt_list)
    gt_mask_list = np.asarray(gt_mask_list)
    gt_mask_list = (gt_mask_list > 0).astype(np.int_)

    segment_scores = np.array(segment_scores)
    image_scores = np.array(image_scores)

    segment_scores = (segment_scores - segment_scores.min()) / (segment_scores.max() - segment_scores.min())
    image_scores = (image_scores - image_scores.min()) / (image_scores.max() - image_scores.min())

    img_roc_auc_det = roc_auc_score(gt_list, image_scores)
    print(f'{args.obj} AUC : {round(img_roc_auc_det, 4)}')

    seg_roc_auc = roc_auc_score(gt_mask_list.flatten(), segment_scores.flatten())
    print(f'{args.obj} pAUC : {round(seg_roc_auc, 4)}')
    return seg_roc_auc + img_roc_auc_det


if __name__ == '__main__':
    main()

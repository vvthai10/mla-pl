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
from CLIP.prompt_ensemble import AnomalyCLIP_PromptLearner
from PIL import Image
from sklearn.metrics import precision_recall_curve
from loss import FocalLoss, BinaryDiceLoss
from utils import augment, encode_text_with_prompt_ensemble
from prompt import REAL_NAME
import torch.optim as optim

import warnings
warnings.filterwarnings("ignore")

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")

CLASS_INDEX = {'Brain':3, 'Liver':2, 'Retina_RESC':1, 'Retina_OCT2017':-1, 'Chest':-2, 'Histopathology':-3} #
CLASS_INDEX_INV = {3:'Brain', 2:'Liver', 1:'Retina_RESC', -1:'Retina_OCT2017', -2:'Chest', -3:'Histopathology'} # ,


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
    parser.add_argument("--depth", type=int, default=9, help="image size")
    parser.add_argument("--n_ctx", type=int, default=12, help="zero shot")
    parser.add_argument("--t_n_ctx", type=int, default=4, help="zero shot")
    parser.add_argument('--obj', type=str, default='Retina_RESC')
    parser.add_argument('--data_path', type=str, default='./data/')
    parser.add_argument('--ckpt_path', type=str, default='./ckpt/')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--img_size', type=int, default=240)
    parser.add_argument("--epoch", type=int, default=50, help="epochs")
    parser.add_argument("--learning_rate", type=float, default=0.01, help="learning rate")
    parser.add_argument("--features_list", type=int, nargs="+", default=[6, 12, 18, 24], help="features used")
    parser.add_argument('--seed', type=int, default=111)
    args = parser.parse_args()

    setup_seed(args.seed)

    AnomalyCLIP_parameters = {"Prompt_length": args.n_ctx, "learnabel_text_embedding_depth": args.depth,
                              "learnabel_text_embedding_length": args.t_n_ctx}
    
    # fixed feature extractor
    clip_model = create_model(model_name=args.model_name, img_size=args.img_size, device=device, pretrained=args.pretrain, require_pretrained=True)
    clip_model.eval()

    model = CLIP_Inplanted(clip_model=clip_model, features=args.features_list, design_details = AnomalyCLIP_parameters).to(device)
    model.eval()

    for name, param in model.named_parameters():
        param.requires_grad = True

    prompt_learner = AnomalyCLIP_PromptLearner(model.to("cpu"), AnomalyCLIP_parameters)
    prompt_learner.to(device)

    model.to(device)

    # optimizer for only adapters
    text_optimizer = torch.optim.Adam(list(prompt_learner.parameters()), lr=args.learning_rate, betas=(0.5, 0.999))
    seg_optimizer = torch.optim.Adam(list(model.seg_adapters.parameters()), lr=args.learning_rate, betas=(0.5, 0.999))
    det_optimizer = torch.optim.Adam(list(model.det_adapters.parameters()), lr=args.learning_rate, betas=(0.5, 0.999))

    # Scheduler
    text_scheduler = optim.lr_scheduler.ReduceLROnPlateau(text_optimizer, mode='min', factor=0.1, patience=5,
                                                          threshold=0.0095)
    seg_scheduler = optim.lr_scheduler.ReduceLROnPlateau(seg_optimizer, mode='min', factor=0.1, patience=5,
                                                         threshold=0.0095)
    det_scheduler = optim.lr_scheduler.ReduceLROnPlateau(det_optimizer, mode='min', factor=0.1, patience=5,
                                                         threshold=0.0095)

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

    # text_feature_list = [0]
    # # text prompt
    # with torch.cuda.amp.autocast(), torch.no_grad():
    #     for i in [1,2,3]: #,-3,-2,-1
    #         text_feature = encode_text_with_prompt_ensemble(clip_model, REAL_NAME[CLASS_INDEX_INV[i]], device)
    #         text_feature_list.append(text_feature)

    save_score = 0.0
    model.eval()
    prompt_learner.train()
    for epoch in range(args.epoch):
        print('epoch', epoch, ':')
        if epoch != 0:
            print("\ttest:")
            score = test(args, model, prompt_learner, test_loader)
            if score >= save_score:
                save_score = score
                ckp_path = f'{args.ckpt_path}/zero-shot/{args.obj}_epoch_{epoch}.pth'
                torch.save({'seg_adapters': model.seg_adapters.state_dict(),
                            'det_adapters': model.det_adapters.state_dict(),
                            "prompt_learner": prompt_learner.state_dict()},
                           ckp_path)
                print(f'best epoch found: epoch {epoch} ')
            print('\n')

        loss_list = []
        for (image, image_label, mask, seg_idx) in tqdm(train_loader):
            image = image.squeeze(0).to(device)
            image_label = image_label.squeeze(0).to(device)

            seg_idx = seg_idx.item()

            with torch.cuda.amp.autocast():
                image_features, seg_patch_tokens, det_patch_tokens = model.encode_image_learn(image)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                seg_patch_tokens = [p[:, 1:, :] for p in seg_patch_tokens]
                det_patch_tokens = [p[:, 1:, :] for p in det_patch_tokens]

                prompts, tokenized_prompts, compound_prompts_text = prompt_learner(cls_id=None)
                text_features = model.encode_text_learn(prompts, tokenized_prompts, compound_prompts_text).float()
                text_features = torch.stack(torch.chunk(text_features, dim=0, chunks=2), dim=1)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                text_features_cal = text_features.clone().squeeze(0).t()

                # features level
                text_probs = 100.0 * image_features @ text_features_cal #.permute(0, 2, 1)
                text_probs = torch.softmax(text_probs, dim=-1)
                text_probs_loss = F.cross_entropy(text_probs.squeeze(), image_label.long())

                # image level
                det_loss = 0
                for layer in range(len(det_patch_tokens)):
                    det_patch_tokens[layer] = det_patch_tokens[layer] / det_patch_tokens[layer].norm(dim=-1, keepdim=True)
                    anomaly_map = (100.0 * det_patch_tokens[layer] @ text_features_cal)
                    anomaly_map = torch.softmax(anomaly_map, dim=-1)[:, :, 1]
                    anomaly_score = torch.mean(anomaly_map, dim=-1)
                    det_loss += loss_bce(anomaly_score, image_label)

                # pixel level
                if seg_idx > 0:
                    seg_loss = 0
                    mask = mask.squeeze(0).to(device)
                    mask[mask > 0.5], mask[mask <= 0.5] = 1, 0
                    for layer in range(len(seg_patch_tokens)):
                        seg_patch_tokens[layer] = seg_patch_tokens[layer] / seg_patch_tokens[layer].norm(dim=-1, keepdim=True)
                        # seg_patch_tokens[layer].shape torch.Size([289, 768])
                        # text_feature_list[seg_idx].shape) torch.Size([768, 2])
                        anomaly_map = (100.0 * seg_patch_tokens[layer] @ text_features_cal)
                        B, L, C = anomaly_map.shape
                        H = int(np.sqrt(L))
                        anomaly_map = F.interpolate(
                            anomaly_map.permute(0, 2, 1).view(B, 2, H, H),
                            size=args.img_size,
                            mode='bilinear',
                            align_corners=True)
                        anomaly_map = torch.softmax(anomaly_map, dim=1)
                        seg_loss += loss_focal(anomaly_map, mask)
                        seg_loss += loss_dice(anomaly_map[:, 1, :, :], mask)

                    loss = det_loss + seg_loss # = focal(seg_out, mask) + bce(det_out, y) text_probs_loss +
                    loss.requires_grad_(True)
                    text_optimizer.zero_grad()
                    seg_optimizer.zero_grad()
                    det_optimizer.zero_grad()
                    loss.backward()
                    text_optimizer.step()
                    seg_optimizer.step()
                    det_optimizer.step()
                else:
                    loss = det_loss  # = focal(seg_out, mask) + bce(det_out, y) text_probs_loss +
                    loss.requires_grad_(True)
                    text_optimizer.zero_grad()
                    det_optimizer.zero_grad()
                    loss.backward()
                    text_optimizer.step()
                    det_optimizer.step()

                loss_list.append(loss.item())

        train_dataset.shuffle_dataset()
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=True, **kwargs)

        # logs
        print("Loss: ", np.mean(loss_list))

        text_scheduler.step(np.mean(loss_list))
        seg_scheduler.step(np.mean(loss_list))
        det_scheduler.step(np.mean(loss_list))


def test(args, model, prompt_learner, test_loader):
    gt_list = []
    gt_mask_list = []
    image_scores = []
    segment_scores = []
    
    for (image, y, mask) in tqdm(test_loader):
        image = image.to(device)
        mask[mask > 0.5], mask[mask <= 0.5] = 1, 0

        with torch.no_grad(), torch.cuda.amp.autocast():
            image_features, ori_seg_patch_tokens, ori_det_patch_tokens = model.encode_image_learn(image)
            ori_seg_patch_tokens = [p[0, 1:, :] for p in ori_seg_patch_tokens]
            ori_det_patch_tokens = [p[0, 1:, :] for p in ori_det_patch_tokens]

            prompts, tokenized_prompts, compound_prompts_text = prompt_learner(cls_id=None)
            text_features = model.encode_text_learn(prompts, tokenized_prompts, compound_prompts_text).float()
            text_features = torch.stack(torch.chunk(text_features, dim=0, chunks=2), dim=1)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            text_features = text_features.squeeze(0).t()

            # features level
            # text_probs = image_features @ text_features  # .permute(0, 2, 1)
            # text_probs = text_probs[:, 0, ...] / 0.07
            # text_probs_loss = F.cross_entropy(text_probs.squeeze(), y.squeeze(0).to(device).long())

            # image
            anomaly_score = 0
            patch_tokens = ori_det_patch_tokens.copy()
            for layer in range(len(patch_tokens)):
                patch_tokens[layer] /= patch_tokens[layer].norm(dim=-1, keepdim=True)
                anomaly_map = (100.0 * patch_tokens[layer] @ text_features).unsqueeze(0)
                anomaly_map = torch.softmax(anomaly_map, dim=-1)[:, :, 1]
                anomaly_score += anomaly_map.mean()
            image_scores.append(anomaly_score.cpu())

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
                anomaly_maps.append(anomaly_map.cpu().numpy())
            final_score_map = np.sum(anomaly_maps, axis=0)
            
            gt_mask_list.append(mask.squeeze().cpu().detach().numpy())
            gt_list.extend(y.cpu().detach().numpy())
            segment_scores.append(final_score_map)

    gt_list = np.array(gt_list)
    gt_mask_list = np.asarray(gt_mask_list)
    gt_mask_list = (gt_mask_list>0).astype(np.int_)

    segment_scores = np.array(segment_scores)
    image_scores = np.array(image_scores)

    segment_scores = (segment_scores - segment_scores.min()) / (segment_scores.max() - segment_scores.min())
    image_scores = (image_scores - image_scores.min()) / (image_scores.max() - image_scores.min())

    img_roc_auc_det = roc_auc_score(gt_list, image_scores)
    print(f'{args.obj} AUC : {round(img_roc_auc_det,4)}')

    if CLASS_INDEX[args.obj] > 0:
        seg_roc_auc = roc_auc_score(gt_mask_list.flatten(), segment_scores.flatten())
        print(f'{args.obj} pAUC : {round(seg_roc_auc,4)}')
        return seg_roc_auc + img_roc_auc_det
    else:
        return img_roc_auc_det

if __name__ == '__main__':
    main()



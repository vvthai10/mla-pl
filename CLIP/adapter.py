import os
import argparse
import random
import math
import numpy as np
from torchvision import transforms
import torch
from torch import nn
from torch.nn import functional as F
from PIL import Image



# Residual CLIP Adapter
class ClipAdapter(nn.Module):
    def __init__(self, c_in, bottleneck_1=768, bottleneck_2=384):
        super(ClipAdapter, self).__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(c_in, bottleneck_1, bias=False),
            nn.LeakyReLU(inplace=False),
            nn.Linear(bottleneck_1, bottleneck_2, bias=False),
            nn.LeakyReLU(inplace=False),
            nn.Linear(bottleneck_2, bottleneck_1, bias=False),
            nn.SiLU(inplace=False)
        )
        self.fc2 = nn.Sequential(
            nn.Linear(bottleneck_1, c_in, bias=False),
            nn.LeakyReLU(inplace=False)
        )

    def forward(self, x):
        x = self.fc1(x)
        y = self.fc2(x)
        return x, y

class CLIP_Inplanted(nn.Module):
    def __init__(self, img_size, clip_model, text_feature_list, features):
        super().__init__()
        self.img_size = img_size
        self.transform_x = transforms.Compose([
            transforms.Resize((img_size, img_size), Image.BICUBIC),
            transforms.ToTensor(),
        ])
        self.clip_model = clip_model
        self.image_encoder = clip_model.visual
        self.text_feature_list = text_feature_list
        self.features = features
        self.n_split_handle = len(features) // 2
        self.seg_adapters = nn.ModuleList( [ClipAdapter(1024, bottleneck_1=768, bottleneck_2=384) for i in range(len(features))] )
        self.det_adapters = nn.ModuleList( [ClipAdapter(1024, bottleneck_1=768, bottleneck_2=384) for i in range(len(features))] )

    def forward(self, x, class_index):
        ori_x = x.clone()
        x = self.image_encoder.conv1(x)
        x = x.reshape(x.shape[0], x.shape[1], -1) 
        x = x.permute(0, 2, 1) 

        x = torch.cat([self.image_encoder.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device),x], dim=1)
        x = x + self.image_encoder.positional_embedding.to(x.dtype)

        x = self.image_encoder.patch_dropout(x)
        x = self.image_encoder.ln_pre(x)

        x = x.permute(1, 0, 2)

        attn_out = []
        seg_patch_tokens = []
        det_patch_tokens = []

        for i in range(24):
            if i + 1 == 12:
                x, attn = self.image_encoder.transformer.resblocks[i](x, attn_mask=None)
                attn_out.append(attn)
            else:
                x, attn_map = self.image_encoder.transformer.resblocks[i](x, attn_mask=None)
            if (i + 1) in self.features:
                seg_adapt_med, seg_adapt_out = self.seg_adapters[self.features.index(i+1)](x)
                det_adapt_med, det_adapt_out = self.det_adapters[self.features.index(i+1)](x)

                x = 0.8 * x + 0.1 * seg_adapt_out + 0.1 * det_adapt_out

                seg_patch_tokens.append(seg_adapt_med)
                det_patch_tokens.append(det_adapt_med)
            if i + 1 == 12:
                assert len(seg_patch_tokens) == 2
                clone_seg_patch_tokens = seg_patch_tokens.copy()
                clone_seg_patch_tokens = [clone_seg_patch_tokens[t].permute(1, 0, 2) for t in range(len(clone_seg_patch_tokens))]
                clone_seg_patch_tokens = [p[0, 1:, :] for p in clone_seg_patch_tokens]

                anomaly_maps = []
                for layer in range(len(clone_seg_patch_tokens)):
                    clone_seg_patch_tokens[layer] = clone_seg_patch_tokens[layer] / clone_seg_patch_tokens[layer].norm(dim=-1,keepdim=True)
                    anomaly_map = (100.0 * clone_seg_patch_tokens[layer] @ self.text_feature_list[class_index]).unsqueeze(0)
                    B, L, C = anomaly_map.shape
                    H = int(np.sqrt(L))
                    anomaly_map = F.interpolate(anomaly_map.permute(0, 2, 1).view(B, 2, H, H), size=self.img_size,
                                                mode='bilinear', align_corners=True)
                    anomaly_map = torch.softmax(anomaly_map, dim=1)
                    anomaly_maps.append(anomaly_map[:, 1, :, :])

                batch = ori_x.shape[0]
                second_input = []
                for b in range(batch):
                    b_img = ori_x[b].clone()
                    ori_img = b_img * 255
                    ori_img = ori_img.cpu().numpy().astype(np.uint8)
                    ori_img = np.transpose(ori_img, (1, 2, 0))

                    anomaly_map_combined = anomaly_maps[0]
                    for anomaly_map in anomaly_maps[1:]:
                        anomaly_map_combined = anomaly_map_combined + anomaly_map.clone()
                    anomaly_map_combined = torch.clamp(anomaly_map_combined, min=0, max=1)

                    anomaly_map_combined = anomaly_map_combined.squeeze().cpu().detach().numpy()
                    mask_detected = np.stack((anomaly_map_combined,) * 3, axis=-1)

                    local_image = np.multiply(ori_img, mask_detected).astype(np.uint8)
                    local_image = Image.fromarray(local_image)
                    local_image = self.transform_x(local_image)
                    second_input += [local_image]
                second_input = torch.stack(second_input)
                x = second_input.to(x.device)
                x = self.image_encoder.conv1(x)
                x = x.reshape(x.shape[0], x.shape[1], -1)
                x = x.permute(0, 2, 1)

                x = torch.cat([self.image_encoder.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1],
                                                                                            dtype=x.dtype,
                                                                                            device=x.device), x], dim=1)
                x = x + self.image_encoder.positional_embedding.to(x.dtype)

                x = self.image_encoder.patch_dropout(x)
                x = self.image_encoder.ln_pre(x)

                x = x.permute(1, 0, 2)

        B, C, L = attn_out[0].shape
        H = int(math.sqrt(L-1))
        out_attn = torch.zeros([H, H]).to('cuda')

        for i in range(len(attn_out)):
            out_attn = out_attn + attn_out[i][0, 0, 1:].view(H, H)
        x = x.permute(1, 0, 2)

        seg_patch_tokens = [seg_patch_tokens[t].permute(1, 0, 2) for t in range(len(seg_patch_tokens))]
        det_patch_tokens = [det_patch_tokens[t].permute(1, 0, 2) for t in range(len(det_patch_tokens))]

        pooled, tokens = self.image_encoder._global_pool(x)
        pooled = self.image_encoder.ln_post(pooled)

        if self.image_encoder.proj is not None:
            pooled = pooled @ self.image_encoder.proj

        return pooled, seg_patch_tokens , det_patch_tokens

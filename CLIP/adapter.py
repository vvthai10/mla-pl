import os
import argparse
import random
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from PIL import Image


# Residual CLIP Adapter
class ClipAdapter(nn.Module):
    def __init__(self, c_in, bottleneck=768):
        super(ClipAdapter, self).__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(c_in, bottleneck, bias=False),
            nn.LeakyReLU(inplace=False)
        )
        self.fc2 = nn.Sequential(
            nn.Linear(bottleneck, c_in, bias=False),
            nn.LeakyReLU(inplace=False)
        )

    def forward(self, x):
        x = self.fc1(x)
        y = self.fc2(x)
        return x, y


class ClipAdapterUpper(nn.Module):
    def __init__(self, c_in, bottleneck=768):
        super(ClipAdapterUpper, self).__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(c_in, bottleneck, bias=False),
            nn.LeakyReLU(inplace=False)
        )
        # self.fc2 = nn.Sequential(
        #     nn.Linear(bottleneck, c_in, bias=False),
        #     nn.LeakyReLU(inplace=False)
        # )

    def forward(self, x):
        x = self.fc1(x)
        # y = self.fc2(x)
        return x, None


class CLIP_Inplanted(nn.Module):
    def __init__(self, clip_model, features):
        super().__init__()
        self.clipmodel = clip_model
        self.image_encoder = clip_model.visual
        self.features = features
        self.seg_adapters = nn.ModuleList([ClipAdapter(1024, bottleneck=768) for i in range(len(features))])
        self.det_adapters = nn.ModuleList([ClipAdapter(1024, bottleneck=768) for i in range(len(features))])

    def forward(self, x):
        x = self.image_encoder.conv1(x)
        x = x.reshape(x.shape[0], x.shape[1], -1)
        x = x.permute(0, 2, 1)

        x = torch.cat(
            [self.image_encoder.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype,
                                                                          device=x.device),
             x], dim=1)
        x = x + self.image_encoder.positional_embedding.to(x.dtype)

        x = self.image_encoder.patch_dropout(x)
        x = self.image_encoder.ln_pre(x)

        x = x.permute(1, 0, 2)

        seg_patch_tokens = []
        det_patch_tokens = []
        # image_features = []
        seg_patch_outs = []
        det_patch_outs = []
        # upper_patch_outs = []

        # text_embeddings = torch.cat((text_embeddings[..., 0], text_embeddings[..., 1]), dim=0)

        for i in range(24):
            x = self.image_encoder.transformer.resblocks[i](x)
            if (i + 1) in self.features:
                x_vv, x_ori = x
                seg_adapt_med, seg_adapt_out = self.seg_adapters[self.features.index(i + 1)](x_vv)
                x_vv[0] = x_ori[0]
                det_adapt_med, det_adapt_out = self.det_adapters[self.features.index(i + 1)](x_vv)

                # upper_adapt_in = 0.5*seg_adapt_med + 0.5*det_adapt_med
                # upper_adapt_out, _ = self.upper_adapters[self.features.index(i+1)](upper_adapt_in)

                # x_ori = 0.8 * x_ori + 0.1 * seg_adapt_out + 0.1 * det_adapt_out

                seg_patch_tokens.append(seg_adapt_med)
                det_patch_tokens.append(det_adapt_med)

                # x_ori = x_ori.permute(1, 0, 2)
                # x_ori = self.ln_post(x_ori[:, 0, :])
                # x_ori = x_ori @ self.image_encoder.proj
                # image_features.append(x_ori)

                # upper_adapt_out = upper_adapt_out.permute(1, 0, 2)
                # upper_adapt_out, _ = self.image_encoder._global_pool(upper_adapt_out)
                # upper_adapt_out = self.image_encoder.ln_post(upper_adapt_out)
                # upper_adapt_out = upper_adapt_out @ self.image_encoder.proj
                # upper_patch_outs.append(upper_adapt_out)

                seg_adapt_out = seg_adapt_out.permute(1, 0, 2)
                # seg_adapt_out, _ = self.image_encoder._global_pool(seg_adapt_out)
                seg_adapt_out = self.image_encoder.ln_post(seg_adapt_out[:, 0, :])
                seg_adapt_out = seg_adapt_out @ self.image_encoder.proj
                seg_patch_outs.append(seg_adapt_out)

                det_adapt_out = det_adapt_out.permute(1, 0, 2)
                # det_adapt_out, _ = self.image_encoder._global_pool(det_adapt_out)
                det_adapt_out = self.image_encoder.ln_post(det_adapt_out[:, 0, :])
                det_adapt_out = det_adapt_out @ self.image_encoder.proj
                det_patch_outs.append(det_adapt_out)

        seg_patch_tokens = [seg_patch_tokens[t].permute(1, 0, 2) for t in range(len(seg_patch_tokens))]
        det_patch_tokens = [det_patch_tokens[t].permute(1, 0, 2) for t in range(len(det_patch_tokens))]

        x = x[1]
        x = x.permute(1, 0, 2)

        pooled, tokens = self.image_encoder._global_pool(x)
        pooled = self.image_encoder.ln_post(pooled)

        if self.image_encoder.proj is not None:
            pooled = pooled @ self.image_encoder.proj

        return pooled, [seg_patch_tokens, det_patch_tokens], [seg_patch_outs, det_patch_outs]





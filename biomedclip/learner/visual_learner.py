import os
import argparse
import random
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from PIL import Image


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


class CLIP_Inplanted(nn.Module):
    def __init__(self, biomedclip, features):
        super().__init__()
        self.biomedclip = biomedclip
        self.image_encoder = biomedclip.visual.trunk
        self.features = features
        self.seg_adapters = nn.ModuleList([ClipAdapter(768, bottleneck=512) for i in range(len(features))])
        self.det_adapters = nn.ModuleList([ClipAdapter(768, bottleneck=512) for i in range(len(features))])

    def forward(self, x):
        x = self.image_encoder.patch_embed(x)
        x = self.image_encoder._pos_embed(x)
        x = self.image_encoder.norm_pre(x)

        seg_patch_tokens = []
        det_patch_tokens = []

        for i, block in enumerate(self.image_encoder.blocks):
            x = block(x)
            if (i + 1) in self.features:
                seg_adapt_med, seg_adapt_out = self.seg_adapters[self.features.index(i + 1)](x)
                det_adapt_med, det_adapt_out = self.det_adapters[self.features.index(i + 1)](x)

                x = 0.8 * x + 0.1 * seg_adapt_out + 0.1 * det_adapt_out

                seg_patch_tokens.append(seg_adapt_med)
                det_patch_tokens.append(det_adapt_med)

        # seg_patch_tokens = [seg_patch_tokens[t].permute(1, 0, 2) for t in range(len(seg_patch_tokens))]
        # det_patch_tokens = [det_patch_tokens[t].permute(1, 0, 2) for t in range(len(det_patch_tokens))]

        x = self.image_encoder.norm(x)

        if self.image_encoder.global_pool:
            x = (
                x[:, self.image_encoder.num_prefix_tokens:].mean(dim=1)
                if self.image_encoder.global_pool == "avg"
                else x[:, 0]
            )
        x = self.image_encoder.fc_norm(x)
        x = self.image_encoder.head(x)

        # Linear Projection: 768 -> 512
        pooled = self.biomedclip.visual.head(x)

        return pooled, seg_patch_tokens, det_patch_tokens

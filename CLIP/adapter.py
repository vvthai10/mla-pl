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

        
class CLIP_Inplanted(nn.Module):
    def __init__(self, clip_model, features):
        super().__init__()
        self.clipmodel = clip_model
        self.image_encoder = clip_model.visual
        self.features = features
        self.seg_adapters = nn.ModuleList( [ClipAdapter(1024, bottleneck=768) for i in range(len(features))] )
        self.det_adapters = nn.ModuleList( [ClipAdapter(768, bottleneck=384) for i in range(len(features))] )


    def forward(self, x, text_embeddings):
        x = self.image_encoder.conv1(x)
        x = x.reshape(x.shape[0], x.shape[1], -1) 
        x = x.permute(0, 2, 1) 

        x = torch.cat(
            [self.image_encoder.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device),
             x], dim=1)
        x = x + self.image_encoder.positional_embedding.to(x.dtype)

        x = self.image_encoder.patch_dropout(x)
        x = self.image_encoder.ln_pre(x)

        x = x.permute(1, 0, 2)

        seg_patch_tokens = []
        det_patch_tokens = []

        text_embeddings = torch.cat((text_embeddings[..., 0], text_embeddings[..., 1]), dim=0)

        for i in range(24):
            x = self.image_encoder.transformer.resblocks[i](x)
            if (i + 1) in self.features:
                seg_adapt_med, seg_adapt_out = self.seg_adapters[self.features.index(i+1)](0.5*x[0] + 0.5*x[1])
                det_adapt_med, det_adapt_out = self.det_adapters[self.features.index(i+1)](x[1])
                # x[1] = 0.8 * x[1] + 0.1 * seg_adapt_out + 0.1 * det_adapt_out
                seg_adapt_out = F.linear(seg_adapt_out, text_embeddings.t())
                seg_adapt_med = 0.5*seg_adapt_med + 0.5*seg_adapt_out
                # det_adapt_out = F.linear(det_adapt_out, text_embeddings.t())
                # det_adapt_med = 0.5 * det_adapt_med + 0.5 * det_adapt_out
                seg_patch_tokens.append(seg_adapt_med)
                # det_patch_tokens.append(det_adapt_med)

                det_in = x[1]
                det_in = det_in.permute(1, 0, 2)
                det_in = det_in @ self.image_encoder.proj
                det_in = det_in.permute(1, 0, 2)
                _, det_adapt_out = self.det_adapters[self.features.index(i + 1)](det_in)
                det_patch_tokens.append(det_adapt_out)

        seg_patch_tokens = [seg_patch_tokens[t].permute(1, 0, 2) for t in range(len(seg_patch_tokens))]
        det_patch_tokens = [det_patch_tokens[t].permute(1, 0, 2) for t in range(len(det_patch_tokens))]

        x = x[1]
        x = x.permute(1, 0, 2)

        pooled, tokens = self.image_encoder._global_pool(x)
        pooled = self.image_encoder.ln_post(pooled)

        if self.image_encoder.proj is not None:
            pooled = pooled @ self.image_encoder.proj

        return pooled, seg_patch_tokens, det_patch_tokens





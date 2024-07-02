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

class ClipAdapterDet(nn.Module):
    def __init__(self, c_in, bottleneck=768):
        super(ClipAdapterDet, self).__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(c_in, bottleneck, bias=False),
            nn.LeakyReLU(inplace=False)
        )

    def forward(self, x):
        x = self.fc1(x)
        return x

        
class CLIP_Inplanted(nn.Module):
    def __init__(self, features):
        super().__init__()
        self.features = features
        self.seg_adapters = nn.ModuleList( [ClipAdapter(1024, bottleneck=768) for i in range(len(features))] )
        self.det_adapters = nn.ModuleList( [ClipAdapter(768, bottleneck=384) for i in range(len(features))] )


    def forward(self, seg_tokens, det_tokens, text_embeddings):
        seg_patch_tokens = []
        det_patch_tokens = []
        text_embeddings = torch.cat((text_embeddings[..., 0], text_embeddings[..., 1]), dim=0)

        for i in range(len(seg_tokens)):
            seg_adapt_med, seg_adapt_out = self.seg_adapters[i](seg_tokens[i])
            det_adapt_med = self.det_adapters[i](det_tokens[i])

            seg_adapt_out = F.linear(seg_adapt_out, text_embeddings.t())
            seg_adapt_med = 0.5*seg_adapt_med + 0.5*seg_adapt_out
            seg_patch_tokens.append(seg_adapt_med)

            det_patch_tokens.append(det_adapt_med)

        seg_patch_tokens = [seg_patch_tokens[t].permute(1, 0, 2) for t in range(len(seg_patch_tokens))]
        det_patch_tokens = [det_patch_tokens[t].permute(1, 0, 2) for t in range(len(det_patch_tokens))]

        return seg_patch_tokens, det_patch_tokens





import math


import torch
from torch import nn


# Residual CLIP Adapter
class ClipAdapter(nn.Module):
    def __init__(self, c_in, bottleneck=768):
        super(ClipAdapter, self).__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(c_in, bottleneck, bias=False),
            nn.LeakyReLU(inplace=False),
        )
        self.fc2 = nn.Sequential(
            nn.Linear(bottleneck, c_in, bias=False),
            nn.LeakyReLU(inplace=False),
        )

    def forward(self, x):
        x = self.fc1(x)
        y = self.fc2(x)
        return x, y


class CLIP_Inplanted(nn.Module):
    def __init__(self, clip_model, features, reduce_dim=768, decoder_heads=4):
        super().__init__()
        self.clipmodel = clip_model
        self.image_encoder = clip_model.visual
        self.features = features

        # Segment Adapter
        self.seg_adapters = nn.ModuleList(
            [ClipAdapter(1024, bottleneck=reduce_dim) for i in range(len(features))]
        )

        # Classification Adapter
        self.det_adapters = nn.ModuleList(
            [ClipAdapter(1024, bottleneck=reduce_dim) for i in range(len(features))]
        )

        self.decoder = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(d_model=reduce_dim, nhead=decoder_heads)
                for _ in range(len(self.features))
            ]
        )

    def forward(self, image, text_features):
        x = self.image_encoder.conv1(image)
        x = x.reshape(x.shape[0], x.shape[1], -1)
        x = x.permute(0, 2, 1)

        x = torch.cat(
            [
                self.image_encoder.class_embedding.to(x.dtype)
                + torch.zeros(
                    x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device
                ),
                x,
            ],
            dim=1,
        )
        x = x + self.image_encoder.positional_embedding.to(x.dtype)

        x = self.image_encoder.patch_dropout(x)
        x = self.image_encoder.ln_pre(x)

        x = x.permute(1, 0, 2)

        seg_patch_tokens = []
        det_patch_tokens = []
        attn_out = []
        activations = []

        for i in range(24):

            if i + 1 == 12:
                x, attn = self.image_encoder.transformer.resblocks[i](x)
                attn_out.append(attn)
            else:
                x, _ = self.image_encoder.transformer.resblocks[i](x)

            if (i + 1) in self.features:
                # 290, 1, 256
                seg_adapt_med, seg_adapt_out = self.seg_adapters[
                    self.features.index(i + 1)
                ](x)

                det_adapt_med, det_adapt_out = self.det_adapters[
                    self.features.index(i + 1)
                ](x)


                det_adapt_med = self.forward_decoder(
                    det_adapt_med, text_features, self.features.index(i + 1)
                )

                seg_adapt_med = self.forward_decoder(
                    seg_adapt_med, text_features, self.features.index(i + 1)
                )

                # F^*
                x = 0.8 * x + 0.1 * seg_adapt_out + 0.1 * det_adapt_out

                seg_patch_tokens.append(seg_adapt_med)
                det_patch_tokens.append(det_adapt_med)

            activations.append(x)

        B, C, L = attn_out[0].shape
        H = int(math.sqrt(L - 1))

        out_attn = torch.zeros([H, H]).to("cuda")

        for i in range(len(attn)):
            out_attn = out_attn + attn_out[i][0, 0, 1:].view(H, H)

        x = x.permute(1, 0, 2)

        seg_patch_tokens = [
            seg_patch_tokens[t].permute(1, 0, 2) for t in range(len(seg_patch_tokens))
        ]
        det_patch_tokens = [
            det_patch_tokens[t].permute(1, 0, 2) for t in range(len(det_patch_tokens))
        ]

        pooled, tokens = self.image_encoder._global_pool(x)
        pooled = self.image_encoder.ln_post(pooled)

        if self.image_encoder.proj is not None:
            pooled = pooled @ self.image_encoder.proj

        return pooled, seg_patch_tokens, det_patch_tokens
    def forward_decoder(self, patch_token, text_features, ith):

        patch_token = (self.decoder[ith](patch_token)).permute(1, 0, 2)  # 1, 290, 256
        # print("patch_token 2 shape: ", patch_token.shape)

        patch_token /= patch_token.norm(dim=-1, keepdim=True)  # 1, 290, 256
        # print("patch_token 3 shape: ", patch_token.shape)

        # (1, 290, 256) @ (256, 2) => (1, 290, 256)
        patch_token = (patch_token @ text_features).permute(1, 0, 2) 
        # print("patch_token 4 shape: ", patch_token.shape)
        return patch_token

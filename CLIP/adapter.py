import torch
from torch import nn
from torch.nn import functional as F

class TextAdapter(nn.Module):
    def __init__(self, text_embeddings, label=None, beta=5.5):
        super(TextAdapter, self).__init__()
        # self.text_layer = nn.Linear(text_embeddings.shape[1], text_embeddings.shape[0], bias=False).to(text_embeddings.device)
        # self.text_layer.weight = nn.Parameter(text_embeddings)
        text_embeddings = torch.cat(
            (text_embeddings[..., 0], text_embeddings[..., 1]), dim=0
        )
        print("Text embeddings: ", text_embeddings.shape)
        self.ad = torch.nn.Linear(text_embeddings.shape[1], text_embeddings.shape[0])
        # self.n =  nn.Linear(text_embeddings.shape[1], text_embeddings.shape[0], bias=False).to(text_embeddings.device)
        self.text_embeddings = text_embeddings
        # self.weights = nn.Parameter(text_embeddings)
        # self.label = F.one_hot(label.to(torch.int64)).float()
        self.noise_level = 1
        self.mask_ratio = 0.25
        self.beta = beta

    # def init_parameter(self,):
    # self.ad.weight.data = self.text_embeddings
    # self.weights.data = self.text_embeddings

    def adapter(self, img):
        img = img / img.norm(dim=-1, keepdim=True)

        affinity = self.ad(
            img
        )
        affinity = torch.tanh(affinity)
        output = F.linear(affinity, self.text_embeddings.t())
        return output

    def mask_aug(self, true_feats):
        N, H, W, C = true_feats.shape

        ids_noise = torch.rand(N, H * W, device=true_feats.device)
        ids_shuffle = torch.argsort(ids_noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        len_mask = int(H * W * self.mask_ratio)

        noise = torch.normal(0, 0.05 * 1.1**2, true_feats.shape).to(true_feats.device)
        fake_feats = [true_feats]
        noise_masks = []
        for i in range(int(1 / self.mask_ratio)):
            mask = torch.zeros([N, H * W], device=true_feats.device)
            if i != int(1 / self.mask_ratio):
                mask[:, i * len_mask : (i + 1) * len_mask] = 1
            else:
                mask[:, i * len_mask :] = 1
            noise_mask = torch.gather(mask, dim=1, index=ids_restore)
            noise_masks.append(noise_mask)
            fake_feat = true_feats + noise * noise_mask.view(N, H, W, 1)
            fake_feats.append(fake_feat)
        return torch.stack(fake_feats, dim=0).view(-1, H, W, C), torch.stack(
            noise_masks, dim=0
        ).view(-1, H, W, 1)

    def aug(self, true_feat):
        N, H, W, C = true_feat.shape
        feat_list = [true_feat]
        for n in range(self.noise_level):
            noise = torch.normal(0, 0.05 * 1.1 ** (n + 1), true_feat.shape).to(
                true_feat.device
            )
            fake_feat = true_feat + noise
            feat_list.append(fake_feat)
        return torch.stack(feat_list, dim=0).view(-1, H, W, C)

    def forward(self, x, is_test=False, scale=0.1):
        if not is_test:
            x = self.aug(x)
        if len(x.shape) == 4:
            N, H, W, C = x.shape
            x = 0.5 * x.view(N, H * W, C) + 0.5 * self.adapter(x.view(N, H * W, C))
            x = x.view(N, H, W, C)
        else:
            x = 0.5 * x + 0.5 * self.adapter(x)
        return x


# Residual CLIP Adapter
class VisualAdapter(nn.Module):
    def __init__(self, c_in, bottleneck=768):
        super(VisualAdapter, self).__init__()
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
    def __init__(
        self,
        clip_model,
        features,
        seg_reduce_dim=128,
        det_reduce_dim=768,
        decoder_heads=4,
        extra_blocks=0
    ):
        super().__init__()
        self.clipmodel = clip_model
        self.image_encoder = clip_model.visual
        self.features = features

        # Segment Adapter
        self.seg_adapters = nn.ModuleList(
            [VisualAdapter(1024, bottleneck=seg_reduce_dim) for i in range(len(features))]
        )

        # Classification Adapter
        self.det_adapters = nn.ModuleList(
            [VisualAdapter(1024, bottleneck=det_reduce_dim) for i in range(len(features))]
        )

        self.decoder = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(d_model=seg_reduce_dim, nhead=decoder_heads)
                for _ in range(len(self.features))
            ]
        )

        self.extra_blocks = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(d_model=seg_reduce_dim, nhead=decoder_heads)
                for _ in range(extra_blocks)
            ]
        )

        self.text_proj = nn.Linear(768, seg_reduce_dim, device="cuda:0")

    def forward(self, image):
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

        for i in range(24):
            x = self.image_encoder.transformer.resblocks[i](x)

            if (i + 1) in self.features:

                seg_adapt_med, seg_adapt_out = self.seg_adapters[
                    self.features.index(i + 1)
                ](x[1])

                det_adapt_med, det_adapt_out = self.det_adapters[
                    self.features.index(i + 1)
                ](x[1])

                x[1] = 0.8 * x[1] + 0.1 * seg_adapt_out + 0.1 * det_adapt_out

                seg_patch_tokens.append(seg_adapt_med)
                det_patch_tokens.append(det_adapt_med)

        seg_patch_tokens = [
            seg_patch_tokens[t].permute(1, 0, 2) for t in range(len(seg_patch_tokens))
        ]

        det_patch_tokens = [
            det_patch_tokens[t].permute(1, 0, 2) for t in range(len(det_patch_tokens))
        ]

        return None, seg_patch_tokens, det_patch_tokens

    def decode(self, patch_tokens, text_features, ith):

        text = text_features.permute(1, 0)
        text = self.text_proj(text)
        text = text.permute(1, 0)  # [128, 2]

        x = patch_tokens  # [2, 290, 128]
        x = x.permute(1, 0, 2)
        x = self.decoder[ith](x)  # [290, 2, 128]
        x = x.permute(1, 0, 2)

        patch = x[:, 1:, :]  # [2, 289, 128]
        patch = patch / patch.norm(dim=-1, keepdim=True)  # [2, 289, 128]

        # [2, 289, 128] @ [128, 2]
        anomaly_map = 100 * patch @ text  # [2, 289, 2]
        return x, anomaly_map

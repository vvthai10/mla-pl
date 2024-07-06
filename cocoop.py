import os.path as osp

import torch
import torch.nn as nn

from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast
from collections import OrderedDict
from CLIP.tokenizer import SimpleTokenizer,tokenize


class TextEncoder(nn.Module):
    def __init__(self, clip_model):

        super().__init__()

        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection

    def forward(self, prompts, tokenized_prompts):

        x = prompts + self.positional_embedding
        x = x.permute(1, 0, 2)  # NLD -> LND
        x,_,_ = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection
        return x


class PromptLearner(nn.Module):
    def __init__(self,
                 batch_size,
                 prompts,
                 n_ctx, # prompt max len
                 CSC, # True or False multi prompt
                 class_token_position, # cls position
                 clip_model):

        super().__init__()
        device = "cuda"

        ctx_dim = clip_model.ln_final.weight.shape[0] #

        self.ctx={}

        for cls in prompts:
            for position in class_token_position:
                if CSC:
                    ctx_vectors = torch.empty(len(prompts[cls]), n_ctx, ctx_dim).to(device)
                else:
                    ctx_vectors = torch.empty(n_ctx, ctx_dim).to(device)
                nn.init.normal_(ctx_vectors, std=0.02)
                self.ctx['{}_{}'.format(cls,position)]=nn.Parameter(ctx_vectors,requires_grad=True)

        self.ctx = nn.ParameterDict(self.ctx)  # to be optimized

        prompt_prefix = " ".join(["X"] * n_ctx)

        _tokenizer = SimpleTokenizer()

        prompts_split={cls: [prompt.replace("_", " ")  for prompt in prompts[cls]] for cls in prompts}

        prompts_lens = {cls: [len(_tokenizer.encode(prompt)) for prompt in prompts_split[cls]] for cls in prompts_split}

        prompts_learnable_tokens = {cls: [prompt_prefix + " " + prompt + "." for prompt in prompts_split[cls]] for cls in prompts_split}

        tokenized_prompts = {cls: torch.cat([tokenize(prompt) for prompt in prompts_learnable_tokens[cls]]).to(device) for cls in prompts_learnable_tokens}

        with torch.no_grad():
            embeddings = {cls: clip_model.token_embedding(tokenized_prompts[cls]) for cls in tokenized_prompts}

        self.register_embeddings = {}

        for cls in embeddings:
            self.register_embeddings['{}_token_prefix'.format(cls)]=embeddings[cls][:, :1, :]
            self.register_embeddings['{}_token_suffix'.format(cls)]=embeddings[cls][:, 1 + n_ctx :, :]

        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts
        self.prompts_lens = prompts_lens
        self.class_token_position = class_token_position

        vis_dim = clip_model.visual.output_dim
        self.meta_net = nn.Sequential(OrderedDict([
            ("linear1", nn.Linear(vis_dim, vis_dim // 16)),
            ("relu", nn.ReLU(inplace=True)),
            ("linear2", nn.Linear(vis_dim // 16, ctx_dim))
        ]))
        # self.meta_nets = nn.ModuleList([nn.Sequential(OrderedDict([
        #     ("linear1", nn.Linear(vis_dim, vis_dim // 16)),
        #     ("relu", nn.ReLU(inplace=True)),
        #     ("linear2", nn.Linear(vis_dim // 16, ctx_dim))
        # ])) for i in range(4)])

    def forward(self, seg_patch_outs, det_patch_outs):
        image_features = None
        for i in range(len(seg_patch_outs)):
            seg_patch_outs[i] = seg_patch_outs[i] / seg_patch_outs[i].norm(dim=-1, keepdim=True)
            det_patch_outs[i] = det_patch_outs[i] / det_patch_outs[i].norm(dim=-1, keepdim=True)
            each = 0.5 * seg_patch_outs[i] + 0.5 * det_patch_outs[i]

            if image_features is None:
                image_features = each
            else:
                image_features += each
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        bias = self.meta_net(image_features)  # (batch, ctx_dim)
        bias = torch.sum(bias, dim=0, keepdim=True)
        bias = bias / bias.norm(dim=-1, keepdim=True)
        bias = bias.unsqueeze(1)  # (batch, 1, ctx_dim)
        # ctx_shifted = ctx + bias  # (batch, n_ctx, ctx_dim)

        cls_prompts = {}
        for cls in self.tokenized_prompts:

            prefix = self.register_embeddings['{}_token_prefix'.format(cls)]
            suffix = self.register_embeddings['{}_token_suffix'.format(cls)]

            cls_prompts[cls] = []

            for position in self.class_token_position:
                ctx = self.ctx['{}_{}'.format(cls,position)]
                if ctx.dim() == 2:
                    ctx = ctx.unsqueeze(0).expand(len(self.prompts_lens[cls]), -1, -1)
                ctx = ctx + bias
                # if position == "end":
                prompts = torch.cat(
                    [
                        prefix,  # (n_cls, 1, dim)
                        ctx,     # (n_cls, n_ctx, dim)
                        suffix,  # (n_cls, *, dim)
                    ],
                    dim=1,
                )

                cls_prompts[cls].append(prompts)
            cls_prompts[cls] = torch.cat(cls_prompts[cls],dim=0)
        return cls_prompts


class PromptMaker(nn.Module):

    def __init__(self,
                 batch_size,
                 prompts,
                 clip_model,
                 n_ctx: int=8,  # prompt max len
                 CSC: bool= True,  # True or False multi prompt
                 class_token_position: list=['end'],  # cls position
                 ):

        super().__init__()
        assert 'normal' in prompts and 'abnormal' in prompts

        for position in class_token_position:
            assert  position in ['end','middle','front']

        self.prompt_learner = PromptLearner(batch_size,prompts, n_ctx, CSC, class_token_position, clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts

        self.class_token_position = class_token_position
        self.text_encoder = TextEncoder(clip_model)

    def forward(self, seg_patch_outs, det_patch_outs):
        prompts = self.prompt_learner(seg_patch_outs, det_patch_outs)
        tokenized_prompts = self.tokenized_prompts
        text_features=[]

        for cls in prompts:
            class_embedding = self.text_encoder(prompts[cls], tokenized_prompts[cls].repeat(len(self.class_token_position),1))
            class_embedding = class_embedding.mean(dim=0)
            class_embedding = class_embedding / class_embedding.norm()
            text_features.append(class_embedding)
        text_features = torch.stack(text_features, dim=1)
        return text_features #(768,2)
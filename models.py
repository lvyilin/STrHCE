"""
The ViT/DeiT implementation refers to https://github.com/rwightman/pytorch-image-models/blob/v0.3.4/timm/models/vision_transformer.py
"""
import logging
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from functools import partial
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.helpers import load_pretrained
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.registry import register_model

_logger = logging.getLogger(__name__)


def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': .9, 'interpolation': 'bicubic',
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD,
        'first_conv': 'patch_embed.proj', 'classifier': 'head',
        **kwargs
    }


default_cfgs = {
    # patch models (my experiments)
    'vit_small_patch16_224': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/vit_small_p16_224-15ec54c9.pth',
    ),

    # patch models (weights ported from official Google JAX impl)
    'vit_base_patch16_224': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_p16_224-80ecf9dd.pth',
        mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5),
    ),
    'vit_base_patch32_224': _cfg(
        url='',  # no official model weights for this combo, only for in21k
        mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    'vit_base_patch16_384': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_p16_384-83fb41ba.pth',
        input_size=(3, 384, 384), mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), crop_pct=1.0),
    'vit_base_patch32_384': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_p32_384-830016f5.pth',
        input_size=(3, 384, 384), mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), crop_pct=1.0),
    'vit_large_patch16_224': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_large_p16_224-4ee7a4dc.pth',
        mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    'vit_large_patch32_224': _cfg(
        url='',  # no official model weights for this combo, only for in21k
        mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    'vit_large_patch16_384': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_large_p16_384-b3be5167.pth',
        input_size=(3, 384, 384), mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), crop_pct=1.0),
    'vit_large_patch32_384': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_large_p32_384-9b920ba8.pth',
        input_size=(3, 384, 384), mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), crop_pct=1.0),

    # patch models, imagenet21k (weights ported from official Google JAX impl)
    'vit_base_patch16_224_in21k': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_patch16_224_in21k-e5005f0a.pth',
        num_classes=21843, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    'vit_base_patch32_224_in21k': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_patch32_224_in21k-8db57226.pth',
        num_classes=21843, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    'vit_large_patch16_224_in21k': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_large_patch16_224_in21k-606da67d.pth',
        num_classes=21843, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    'vit_large_patch32_224_in21k': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_large_patch32_224_in21k-9046d2e7.pth',
        num_classes=21843, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    'vit_huge_patch14_224_in21k': _cfg(
        url='',  # FIXME I have weights for this but > 2GB limit for github release binaries
        num_classes=21843, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),

    # hybrid models (weights ported from official Google JAX impl)
    'vit_base_resnet50_224_in21k': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_resnet50_224_in21k-6f7c7740.pth',
        num_classes=21843, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), crop_pct=0.9,
        first_conv='patch_embed.backbone.stem.conv'),
    'vit_base_resnet50_384': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_resnet50_384-9fd3c705.pth',
        input_size=(3, 384, 384), mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), crop_pct=1.0,
        first_conv='patch_embed.backbone.stem.conv'),

    # hybrid models (my experiments)
    'vit_small_resnet26d_224': _cfg(),
    'vit_small_resnet50d_s3_224': _cfg(),
    'vit_base_resnet26d_224': _cfg(),
    'vit_base_resnet50d_224': _cfg(),

    # deit models (FB weights)
    'vit_deit_tiny_patch16_224': _cfg(
        url='https://dl.fbaipublicfiles.com/deit/deit_tiny_patch16_224-a1311bcf.pth'),
    'vit_deit_small_patch16_224': _cfg(
        url='https://dl.fbaipublicfiles.com/deit/deit_small_patch16_224-cd65a155.pth'),
    'vit_deit_base_patch16_224': _cfg(
        url='https://dl.fbaipublicfiles.com/deit/deit_base_patch16_224-b5f2ef4d.pth', ),
    'vit_deit_base_patch16_384': _cfg(
        url='https://dl.fbaipublicfiles.com/deit/deit_base_patch16_384-8de9b5d1.pth',
        input_size=(3, 384, 384), crop_pct=1.0),
    'vit_deit_tiny_distilled_patch16_224': _cfg(
        url='https://dl.fbaipublicfiles.com/deit/deit_tiny_distilled_patch16_224-b40b3cf7.pth'),
    'vit_deit_small_distilled_patch16_224': _cfg(
        url='https://dl.fbaipublicfiles.com/deit/deit_small_distilled_patch16_224-649709d9.pth'),
    'vit_deit_base_distilled_patch16_224': _cfg(
        url='https://dl.fbaipublicfiles.com/deit/deit_base_distilled_patch16_224-df68dfff.pth', ),
    'vit_deit_base_distilled_patch16_384': _cfg(
        url='https://dl.fbaipublicfiles.com/deit/deit_base_distilled_patch16_384-d0272ac0.pth',
        input_size=(3, 384, 384), crop_pct=1.0),
}


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        # assert H == self.img_size[0] and W == self.img_size[1], \
        #     f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


class OverlapPatchEmbed(nn.Module):
    def __init__(self, overlap, proj, img_size=224, patch_size=16):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = ((img_size - overlap) // (patch_size - overlap)) ** 2
        self.stride = patch_size - overlap
        self.proj = proj

    def forward(self, x):
        x = F.conv2d(x, self.proj.weight, self.proj.bias, self.stride,
                     self.proj.padding, self.proj.dilation, self.proj.groups)
        # FIXME look at relaxing size constraints
        # assert H == self.img_size[0] and W == self.img_size[1], \
        #     f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = x.flatten(2).transpose(1, 2)
        return x


class VisionTransformer(nn.Module):
    """ Vision Transformer

    A PyTorch impl of : `An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale`  -
        https://arxiv.org/abs/2010.11929
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=True, qk_scale=None, representation_size=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., hybrid_backbone=None, norm_layer=None, overlap=4):
        """
        Args:
            img_size (int, tuple): input image size
            patch_size (int, tuple): patch size
            in_chans (int): number of input channels
            num_classes (int): number of classes for classification head
            embed_dim (int): embedding dimension
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            qk_scale (float): override default qk scale of head_dim ** -0.5 if set
            representation_size (Optional[int]): enable and set representation layer (pre-logits) to this value if set
            drop_rate (float): dropout rate
            attn_drop_rate (float): attention dropout rate
            drop_path_rate (float): stochastic depth rate
            hybrid_backbone (nn.Module): CNN backbone to use in-place of PatchEmbed module
            norm_layer: (nn.Module): normalization layer
        """
        super().__init__()
        self.num_classes = num_classes
        self.patch_size = patch_size
        self.img_size = img_size
        self.overlap = overlap
        assert self.overlap < self.patch_size
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)

        if hybrid_backbone is not None:
            raise NotImplementedError
        else:
            self.patch_embed = PatchEmbed(
                img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
            self.overlap_patch_embed = OverlapPatchEmbed(self.overlap, self.patch_embed.proj, img_size=img_size,
                                                         patch_size=patch_size)
        num_patches = self.num_patches = self.patch_embed.num_patches
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.width = int(self.img_size / self.patch_size)
        self.width_new = int((self.img_size - self.patch_size) / (self.patch_size - self.overlap) + 1)
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_embed_new = nn.Parameter(torch.zeros(1, self.width_new * self.width_new + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)

        # Representation layer
        if representation_size:
            self.num_features = representation_size
            self.pre_logits = nn.Sequential(OrderedDict([
                ('fc', nn.Linear(embed_dim, representation_size)),
                ('act', nn.Tanh())
            ]))
        else:
            self.pre_logits = nn.Identity()

        # Classifier head
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

        trunc_normal_(self.pos_embed, std=.02)
        trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)

        self.attentions = []

        for name, module in self.named_modules():
            if 'attn_drop' in name:
                module.register_forward_hook(self.get_attention)

    def get_attention(self, module, input, output):
        self.attentions.append(output.detach())

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token', 'pos_embed_new'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def get_features(self, x, pos_embed):
        B = x.shape[0]
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
        return x

    def forward(self, x):
        self.attentions.clear()
        B = x.shape[0]
        x_orig = x
        x = self.patch_embed(x)
        x = self.get_features(x, self.pos_embed)
        x_cls = self.head(x[:, 0])

        att_gt = rollout_attention(self.attentions, discard_ratio=0.9)[:, 0, 1:]
        att_min = att_gt.min(1, True)[0]
        att_gt = (att_gt - att_min) / (att_gt.max(1, True)[0] - att_min)

        width, width_new = self.width, self.width_new
        att_map = F.interpolate(att_gt.view(B, 1, width, width),
                                size=(width_new, width_new), mode='bilinear').flatten(1)
        att_map = att_map / att_map.sum(1, True)
        selected_idx = weighted_reservoir_sampling(att_map, width * width).sort(1)[0]
        pos_embed_new = self.get_pos_embed_new(selected_idx)
        x_new = self.overlap_patch_embed(x_orig)
        x_new = torch.stack([x_new[i, selected_idx[i]] for i in range(B)])
        x_new = self.get_features(x_new, pos_embed_new)
        x2_cls = self.head(x_new[:, 0])

        return x_cls, att_gt, x2_cls

    def get_pos_embed_new(self, selected_idx):
        posemb_tok, posemb_grid = self.pos_embed_new[:, :1], self.pos_embed_new[0, 1:]
        posemb_grid = posemb_grid[selected_idx]
        posemb_tok = posemb_tok.expand(selected_idx.size(0), -1, -1)
        posemb = torch.cat([posemb_tok, posemb_grid], dim=1)
        return posemb


class DistilledVisionTransformer(VisionTransformer):
    """ Vision Transformer with distillation token.

    Paper: `Training data-efficient image transformers & distillation through attention` -
        https://arxiv.org/abs/2012.12877

    This impl of distilled ViT is taken from https://github.com/facebookresearch/deit
    """

    def __init__(self, num_parent_classes, num_pparent_classes=None, discard_ratio=0.9, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.hier2 = num_pparent_classes is not None
        self.num_parent_classes = num_parent_classes
        self.discard_ratio = discard_ratio
        num_patches = self.patch_embed.num_patches

        self.dist_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        trunc_normal_(self.dist_token, std=.02)

        self.head2 = nn.Linear(self.embed_dim, self.num_parent_classes)
        self.head2.apply(self._init_weights)
        self.num_extra = 3 if self.hier2 else 2
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_extra, self.embed_dim))
        self.pos_embed_new = nn.Parameter(
            torch.zeros(1, self.width_new * self.width_new + self.num_extra, self.embed_dim))
        trunc_normal_(self.pos_embed, std=.02)
        trunc_normal_(self.pos_embed_new, std=.02)
        if self.hier2:
            self.dist_token2 = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
            trunc_normal_(self.dist_token2, std=.02)
            self.head3 = nn.Linear(self.embed_dim, num_pparent_classes)
            self.head3.apply(self._init_weights)

    def get_features(self, x, pos_embed):
        B = x.shape[0]
        cls_tokens = self.cls_token.expand(B, -1, -1)
        dist_token = self.dist_token.expand(B, -1, -1)
        if self.hier2:
            dist_token2 = self.dist_token2.expand(B, -1, -1)
            x = torch.cat((cls_tokens, dist_token, dist_token2, x), dim=1)
        else:
            x = torch.cat((cls_tokens, dist_token, x), dim=1)
        x = x + pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
        return x

    def forward(self, x):
        self.attentions.clear()
        B = x.shape[0]
        x_orig = x
        x = self.patch_embed(x)
        x = self.get_features(x, self.pos_embed)
        x_cls = self.head(x[:, 0])
        x_cls_hier = self.head2(x[:, 1])
        if self.hier2:
            x_cls_hier2 = self.head3(x[:, 2])

        att_gt = rollout_attention(self.attentions, discard_ratio=self.discard_ratio)[:, 0, self.num_extra:]
        att_min = att_gt.min(1, True)[0]
        att_gt = (att_gt - att_min) / (att_gt.max(1, True)[0] - att_min)

        width, width_new = self.width, self.width_new
        att_map = F.interpolate(att_gt.view(B, 1, width, width),
                                size=(width_new, width_new), mode='bilinear').flatten(1)
        att_map = att_map / att_map.sum(1, True)
        x2_cls_list = []
        x2_cls_hier_list = []
        selected_idx = weighted_reservoir_sampling(att_map, width * width).sort(1)[0]

        pos_embed_new = self.get_pos_embed_new(selected_idx)
        x_new = self.overlap_patch_embed(x_orig)
        x_new = torch.stack([x_new[i, selected_idx[i]] for i in range(B)])
        x_new = self.get_features(x_new, pos_embed_new)
        x2_cls = self.head(x_new[:, 0])
        x2_cls_hier = self.head2(x_new[:, 1])
        if self.hier2:
            x2_cls_hier2 = self.head3(x_new[:, 2])
        x2_cls_list.append(x2_cls)
        x2_cls_hier_list.append(x2_cls_hier)
        if self.hier2:
            return x_cls, x_cls_hier, x_cls_hier2, x2_cls, x2_cls_hier, x2_cls_hier2, att_gt
        return x_cls, x_cls_hier, x2_cls, x2_cls_hier, att_gt

    def get_pos_embed_new(self, selected_idx):
        posemb_tok, posemb_grid = self.pos_embed_new[:, :self.num_extra], self.pos_embed_new[0, self.num_extra:]
        posemb_grid = posemb_grid[selected_idx]
        posemb_tok = posemb_tok.expand(selected_idx.size(0), -1, -1)
        posemb = torch.cat([posemb_tok, posemb_grid], dim=1)
        return posemb


def resize_pos_embed(posemb, posemb_new, distilled, num_extra, variant):
    # Rescale the grid of position embeddings when loading from state_dict. Adapted from
    # https://github.com/google-research/vision_transformer/blob/00883dd691c63a6830751563748663526e811cee/vit_jax/checkpoint.py#L224
    _logger.info('Resized position embedding: %s to %s', posemb.shape, posemb_new.shape)
    ntok_new = posemb_new.shape[1]
    if '21k' in variant:
        posemb_tok, posemb_grid = posemb[:, :1], posemb[0, 1:]
        ntok_new -= num_extra
    elif not distilled or 'deit' not in variant:
        posemb_tok, posemb_grid = posemb[:, :1], posemb[0, 1:]
        ntok_new -= 1
    else:
        posemb_tok, posemb_grid = posemb[:, :2], posemb[0, 2:]
        ntok_new -= num_extra
    gs_old = int(math.sqrt(len(posemb_grid)))
    gs_new = int(math.sqrt(ntok_new))
    _logger.info('Position embedding grid-size from %s to %s', gs_old, gs_new)
    posemb_grid = posemb_grid.reshape(1, gs_old, gs_old, -1).permute(0, 3, 1, 2)
    posemb_grid = F.interpolate(posemb_grid, size=(gs_new, gs_new), mode='bilinear')
    posemb_grid = posemb_grid.permute(0, 2, 3, 1).reshape(1, gs_new * gs_new, -1)
    if '21k' in variant or 'deit' not in variant:
        posemb = torch.cat([posemb_tok] * num_extra + [posemb_grid], dim=1)
    elif num_extra == 2:
        posemb = torch.cat([posemb_tok, posemb_grid], dim=1)
    else:
        posemb = torch.cat([posemb_tok, posemb[:, 1:2], posemb_grid], dim=1)

    return posemb


def checkpoint_filter_fn(state_dict, model, distilled, num_extra, variant):
    """ convert patch embedding weight from manual patchify + linear proj to conv"""
    out_dict = {}
    if 'model' in state_dict:
        # For deit models
        state_dict = state_dict['model']
    for k, v in state_dict.items():
        if 'patch_embed.proj.weight' in k and len(v.shape) < 4:
            # For old models that I trained prior to conv based patchification
            O, I, H, W = model.patch_embed.proj.weight.shape
            v = v.reshape(O, -1, H, W)
        elif k == 'pos_embed' and v.shape != model.pos_embed.shape:
            # To resize pos embedding when using model at different size from pretrained weights
            old_v = v
            v = resize_pos_embed(old_v, model.pos_embed, distilled, num_extra, variant)
            out_dict['pos_embed_new'] = resize_pos_embed(old_v, model.pos_embed_new, distilled, num_extra, variant)

        out_dict[k] = v
    if distilled:
        if 'head_dist.weight' in out_dict:
            del out_dict['head_dist.weight']
        if 'head_dist.weight' in out_dict:
            del out_dict['head_dist.bias']
    return out_dict


def _create_vision_transformer(variant, pretrained=False, distilled=False, **kwargs):
    assert distilled
    default_cfg = default_cfgs[variant]
    default_num_classes = default_cfg['num_classes']
    default_img_size = default_cfg['input_size'][-1]

    num_classes = kwargs.pop('num_classes', default_num_classes)
    img_size = kwargs.pop('img_size', default_img_size)
    repr_size = kwargs.pop('representation_size', None)
    if repr_size is not None and num_classes != default_num_classes:
        # Remove representation layer if fine-tuning. This may not always be the desired action,
        # but I feel better than doing nothing by default for fine-tuning. Perhaps a better interface?
        _logger.warning("Removing representation layer for fine-tuning.")
        repr_size = None

    model_cls = DistilledVisionTransformer if distilled else VisionTransformer
    model = model_cls(img_size=img_size, num_classes=num_classes, representation_size=repr_size, **kwargs)
    model.default_cfg = default_cfg

    num_extra = 3 if 'num_pparent_classes' in kwargs else 2
    if pretrained:
        load_pretrained(
            model, num_classes=num_classes, in_chans=kwargs.get('in_chans', 3),
            filter_fn=partial(checkpoint_filter_fn, model=model, distilled=distilled, num_extra=num_extra,
                              variant=variant),
            strict=False)
    return model


@register_model
def vit_small_patch16_224(pretrained=False, **kwargs):
    """ My custom 'small' ViT model. Depth=8, heads=8= mlp_ratio=3."""
    model_kwargs = dict(
        patch_size=16, embed_dim=768, depth=8, num_heads=8, mlp_ratio=3.,
        qkv_bias=False, norm_layer=nn.LayerNorm, **kwargs)
    if pretrained:
        # NOTE my scale was wrong for original weights, leaving this here until I have better ones for this model
        model_kwargs.setdefault('qk_scale', 768 ** -0.5)
    model = _create_vision_transformer('vit_small_patch16_224', pretrained=pretrained, distilled=True, **model_kwargs)
    return model


@register_model
def vit_base_patch16_224(pretrained=False, **kwargs):
    """ ViT-Base (ViT-B/16) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-1k weights fine-tuned from in21k @ 224x224, source https://github.com/google-research/vision_transformer.
    """
    model_kwargs = dict(patch_size=16, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model = _create_vision_transformer('vit_base_patch16_224', pretrained=pretrained, distilled=True, **model_kwargs)
    return model


@register_model
def vit_base_patch16_384(pretrained=False, **kwargs):
    """ ViT-Base model (ViT-B/16) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-1k weights fine-tuned from in21k @ 384x384, source https://github.com/google-research/vision_transformer.
    """
    model_kwargs = dict(patch_size=16, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model = _create_vision_transformer('vit_base_patch16_384', pretrained=pretrained, distilled=True, **model_kwargs)
    return model


@register_model
def vit_base_patch16_224_in21k(pretrained=False, **kwargs):
    """ ViT-Base model (ViT-B/16) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-21k weights @ 224x224, source https://github.com/google-research/vision_transformer.
    """
    model_kwargs = dict(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, representation_size=768, **kwargs)
    model = _create_vision_transformer('vit_base_patch16_224_in21k', pretrained=pretrained, distilled=True,
                                       **model_kwargs)
    return model


@register_model
def vit_deit_small_patch16_224(pretrained=False, **kwargs):
    """ DeiT-small model @ 224x224 from paper (https://arxiv.org/abs/2012.12877).
    ImageNet-1k weights from https://github.com/facebookresearch/deit.
    """
    model_kwargs = dict(patch_size=16, embed_dim=384, depth=12, num_heads=6, **kwargs)
    model = _create_vision_transformer('vit_deit_small_patch16_224', pretrained=pretrained, distilled=True,
                                       **model_kwargs)
    return model


@register_model
def vit_deit_base_patch16_224(pretrained=False, **kwargs):
    """ DeiT base model @ 224x224 from paper (https://arxiv.org/abs/2012.12877).
    ImageNet-1k weights from https://github.com/facebookresearch/deit.
    """
    model_kwargs = dict(patch_size=16, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model = _create_vision_transformer('vit_deit_base_patch16_224', pretrained=pretrained, distilled=True,
                                       **model_kwargs)
    return model


@register_model
def vit_deit_small_distilled_patch16_224(pretrained=False, **kwargs):
    """ DeiT-small distilled model @ 224x224 from paper (https://arxiv.org/abs/2012.12877).
    ImageNet-1k weights from https://github.com/facebookresearch/deit.
    """
    model_kwargs = dict(patch_size=16, embed_dim=384, depth=12, num_heads=6, **kwargs)
    model = _create_vision_transformer(
        'vit_deit_small_distilled_patch16_224', pretrained=pretrained, distilled=True, **model_kwargs)
    return model


@register_model
def vit_deit_base_distilled_patch16_224(pretrained=False, **kwargs):
    """ DeiT-base distilled model @ 224x224 from paper (https://arxiv.org/abs/2012.12877).
    ImageNet-1k weights from https://github.com/facebookresearch/deit.
    """
    model_kwargs = dict(patch_size=16, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model = _create_vision_transformer(
        'vit_deit_base_distilled_patch16_224', pretrained=pretrained, distilled=True, **model_kwargs)
    return model


@register_model
def vit_deit_base_distilled_patch16_384(pretrained=False, **kwargs):
    """ DeiT-base distilled model @ 384x384 from paper (https://arxiv.org/abs/2012.12877).
    ImageNet-1k weights from https://github.com/facebookresearch/deit.
    """
    model_kwargs = dict(patch_size=16, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model = _create_vision_transformer(
        'vit_deit_base_distilled_patch16_384', pretrained=pretrained, distilled=True, **model_kwargs)
    return model


def rollout_attention(attentions, discard_ratio):
    device = attentions[0].device
    result = eye = torch.eye(attentions[0].size(-1), device=device)
    with torch.no_grad():
        for attention in attentions:
            attention_heads_fused = attention.max(axis=1)[0]
            flat = attention_heads_fused.view(attention_heads_fused.size(0), -1)
            remain_num = int(flat.size(-1) * (1 - discard_ratio))
            indices = flat.topk(remain_num, -1, True, False)[1]
            mask = torch.zeros_like(flat, dtype=torch.bool)
            mask.scatter_(1, indices, True)
            mask[:, 0] = True
            mask = mask.logical_not_()
            flat.masked_fill_(mask, 0.0)

            a = attention_heads_fused + eye
            a = a / a.sum(-1, True)

            result = torch.matmul(a, result)
    return result


def weighted_reservoir_sampling(weight, K):
    weight = weight / weight.sum(1, True)
    u = torch.rand_like(weight)
    r = torch.pow(u, weight.reciprocal())
    return r.topk(K, 1, largest=True, sorted=False)[1]


if __name__ == '__main__':
    model = vit_deit_small_distilled_patch16_224(num_parent_classes=200,
                                                 num_pparent_classes=10,
                                                 pretrained=True,
                                                 img_size=448)
    model = model.eval()
    x = model(torch.randn(6, 3, 448, 448))
    print(len(x))
    print(x[0].size())
    print(x[1].size())

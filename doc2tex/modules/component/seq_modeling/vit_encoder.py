import torch.nn as nn
import torch
from torch.nn import functional as F
from modules.component.seq_modeling.vit.utils import trunc_normal_
from modules.component.seq_modeling.vit.vision_transformer import VisionTransformer
from modules.component.seq_modeling.addon_module import *
from modules.component.feature_extractor import (
    ResNet_FeatureExtractor,
    VGG_FeatureExtractor,
)
from modules.component.common.mae_posembed import get_2d_sincos_pos_embed

__all__ = [
    "ViTEncoder",
    "ViTEncoderV2",
    "ViTEncoderV3",
    "TRIGBaseEncoder",
    "create_vit_modeling",
]


class ViTEncoder(VisionTransformer):
    """ """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if kwargs["hybrid_backbone"] is None:
            self.patch_embed = PatchEmbed(
                img_size=kwargs["img_size"],
                in_chans=kwargs["in_chans"],
                patch_size=kwargs["patch_size"],
                embed_dim=kwargs["embed_dim"],
            )
        else:
            self.patch_embed = HybridEmbed(
                backbone=kwargs["hybrid_backbone"],
                img_size=kwargs["img_size"],
                in_chans=kwargs["in_chans"],
                patch_size=kwargs["patch_size"],
                embed_dim=kwargs["embed_dim"],
            )
        num_patches = self.patch_embed.num_patches
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, kwargs["embed_dim"])
        )
        self.emb_height = self.patch_embed.grid_size[0]
        self.emb_width = self.patch_embed.grid_size[1]
        trunc_normal_(self.pos_embed, std=0.02)
        self.apply(self._init_weights)

    def reset_classifier(self, num_classes):
        self.num_classes = num_classes
        self.head = (
            nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        )

    def interpolating_pos_embedding(self, embedding, height, width):
        """
        Source:
        https://github.com/facebookresearch/dino/blob/de9ee3df6cf39fac952ab558447af1fa1365362a/vision_transformer.py#L174
        """
        npatch = embedding.shape[1] - 1
        N = self.pos_embed.shape[1] - 1
        if npatch == N and height == width:
            return self.pos_embed

        class_pos_embedding = self.pos_embed[:, 0]
        patch_pos_embedding = self.pos_embed[:, 1:]
        dim = self.pos_embed.shape[-1]

        h0 = height // self.patch_embed.patch_size[0]
        w0 = width // self.patch_embed.patch_size[1]
        # add a small number to avo_id floating point error
        # https://github.com/facebookresearch/dino/issues/8

        h0 = h0 + 0.1
        w0 = w0 + 0.1

        patch_pos_embedding = nn.functional.interpolate(
            patch_pos_embedding.reshape(
                1, self.emb_height, self.emb_width, dim
            ).permute(0, 3, 1, 2),
            scale_factor=(h0 / self.emb_height, w0 / self.emb_width),
            mode="bicubic",
            align_corners=False,
        )
        assert (
            int(h0) == patch_pos_embedding.shape[-2]
            and int(w0) == patch_pos_embedding.shape[-1]
        )
        patch_pos_embedding = patch_pos_embedding.permute(0, 2, 3, 1).view(1, -1, dim)
        class_pos_embedding = class_pos_embedding.unsqueeze(0)

        return torch.cat((class_pos_embedding, patch_pos_embedding), dim=1)

    def forward_features(self, x):
        B, C, _, _ = x.shape

        x, pad_info, size, interpolating_pos = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(
            B, -1, -1
        )  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)

        if interpolating_pos:
            x = x + self.interpolating_pos_embedding(x, size["height"], size["width"])
        else:
            x = x + self.pos_embed

        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)

        return x, pad_info, size


class TRIGBaseEncoder(ViTEncoder):
    """
    https://arxiv.org/pdf/2111.08314.pdf
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.patch_embed = HybridEmbed1D(
            backbone=kwargs["hybrid_backbone"],
            img_size=kwargs["img_size"],
            in_chans=kwargs["in_chans"],
            patch_size=kwargs["patch_size"],
            embed_dim=kwargs["embed_dim"],
        )
        num_patches = self.patch_embed.num_patches
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, kwargs["embed_dim"])
        )
        self.emb_height = 1
        self.emb_width = self.patch_embed.grid_size[1]
        trunc_normal_(self.pos_embed, std=0.02)
        self.apply(self._init_weights)

    def interpolating_pos_embedding(self, embedding, height, width):
        """
        Source:
        https://github.com/facebookresearch/dino/blob/de9ee3df6cf39fac952ab558447af1fa1365362a/vision_transformer.py#L174
        """
        npatch = embedding.shape[1] - 1
        N = self.pos_embed.shape[1] - 1
        if npatch == N and height == width:
            return self.pos_embed

        class_pos_embedding = self.pos_embed[:, 0]
        patch_pos_embedding = self.pos_embed[:, 1:]
        dim = self.pos_embed.shape[-1]

        w0 = width // self.patch_embed.window_width

        # add a small number to avoid floating point error
        # https://github.com/facebookresearch/dino/issues/8

        w0 = w0 + 0.1

        patch_pos_embedding = nn.functional.interpolate(
            patch_pos_embedding.reshape(
                1, self.emb_height, self.emb_width, dim
            ).permute(0, 3, 1, 2),
            scale_factor=(1, w0 / self.emb_width),
            mode="bicubic",
            align_corners=False,
        )

        assert int(w0) == patch_pos_embedding.shape[-1]
        patch_pos_embedding = patch_pos_embedding.permute(0, 2, 3, 1).view(1, -1, dim)
        class_pos_embedding = class_pos_embedding.unsqueeze(0)

        return torch.cat((class_pos_embedding, patch_pos_embedding), dim=1)

    def forward_features(self, x):
        B, _, _, _ = x.shape
        x, padinfo, size, interpolating_pos = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(
            B, -1, -1
        )  # stole cls_tokens impl from Phil Wang, thanks

        x = torch.cat(
            (cls_tokens, x), dim=1
        )  # cls_tokens is init_embedding in TRIG paper

        if interpolating_pos:
            x = x + self.interpolating_pos_embedding(x, size["height"], size["width"])
        else:
            x = x + self.pos_embed

        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)

        return x, padinfo, size


class ViTEncoderV2(ViTEncoder):
    def forward(self, x):
        B, _, _, _ = x.shape

        x, pad_info, size, _ = self.patch_embed(x)
        _, numpatches, *_ = x.shape
        cls_tokens = self.cls_token.expand(
            B, -1, -1
        )  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)

        x = x + self.pos_embed[:, : (numpatches + 1)]
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)

        return x, pad_info, size


class ViTEncoderV3(ViTEncoder):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if hasattr(self, "pos_embed"):
            del self.pos_embed
        num_patches = self.patch_embed.num_patches
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, kwargs["embed_dim"]), requires_grad=False
        )
        self.initialize_posembed()

    def initialize_posembed(self):
        pos_embed = get_2d_sincos_pos_embed(
            self.pos_embed.shape[-1],
            self.patch_embed.grid_size[0],
            self.patch_embed.grid_size[1],
            cls_token=True,
        )
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

    def forward(self, x):
        B, _, _, _ = x.shape

        x, pad_info, size, _ = self.patch_embed(x)
        _, numpatches, *_ = x.shape

        cls_tokens = self.cls_token.expand(
            B, -1, -1
        )  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)

        x = x + self.pos_embed[:, : (numpatches + 1)]
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)

        return x, pad_info, size


def create_vit_modeling(opt):
    seq_modeling = opt["SequenceModeling"]["params"]
    if seq_modeling["backbone"] is not None:
        if seq_modeling["backbone"]["name"] == "resnet":
            param_kwargs = dict()
            if seq_modeling["backbone"].get("pretrained", None) is not None:
                param_kwargs["pretrained"] = seq_modeling["backbone"]["pretrained"]
            if seq_modeling["backbone"].get("weight_dir", None) is not None:
                param_kwargs["weight_dir"] = seq_modeling["backbone"]["weight_dir"]
            print("kwargs", param_kwargs)

            backbone = ResNet_FeatureExtractor(
                seq_modeling["backbone"]["input_channel"],
                seq_modeling["backbone"]["output_channel"],
                seq_modeling["backbone"]["gcb"],
                **param_kwargs
            )
        elif seq_modeling["backbone"]["name"] == "cnn":
            backbone = None
    else:
        backbone = None
    max_dimension = (
        (opt["imgH"], opt["max_dimension"][1]) if opt["imgH"] else opt["max_dimension"]
    )
    if seq_modeling["patching_style"] == "2d":
        if seq_modeling.get("fix_embed", False):
            encoder = ViTEncoderV3
        else:
            if not seq_modeling.get("interpolate_embed", True):
                encoder = ViTEncoderV2
            else:
                encoder = ViTEncoder
    else:
        encoder = TRIGBaseEncoder

    encoder_seq_modeling = encoder(
        img_size=max_dimension,
        patch_size=seq_modeling["patch_size"],
        in_chans=seq_modeling["input_channel"],
        depth=seq_modeling["depth"],
        num_classes=0,
        embed_dim=seq_modeling["hidden_size"],
        num_heads=seq_modeling["num_heads"],
        hybrid_backbone=backbone,
    )

    return encoder_seq_modeling

import math
import torch.nn as nn
import torch
from torch.nn import functional as F
from timm.models.layers.helpers import to_2tuple
from typing import Tuple, Union, List


class PatchEmbed(nn.Module):
    """Image to Patch Embedding"""

    def __init__(self, img_size=(224, 224), patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        assert isinstance(img_size, tuple)
        patch_size = to_2tuple(patch_size)
        div_h, mod_h = divmod(img_size[0], patch_size[0])
        div_w, mod_w = divmod(img_size[1], patch_size[1])
        self.img_size = (
            patch_size[0] * (div_h + (1 if mod_h > 0 else 0)),
            patch_size[1] * (div_w + (1 if mod_w > 0 else 0)),
        )
        self.grid_size = (
            self.img_size[0] // patch_size[0],
            self.img_size[1] // patch_size[1],
        )
        self.patch_size = patch_size
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.proj = nn.Conv2d(
            in_chans, embed_dim, kernel_size=patch_size, stride=patch_size
        )

    def forward(self, x):
        _, _, H, W = x.shape
        div_h, mod_h = divmod(H, self.patch_size[0])
        div_w, mod_w = divmod(W, self.patch_size[1])
        pad_H = self.patch_size[0] * (div_h + (1 if mod_h > 0 else 0)) - H
        pad_W = self.patch_size[1] * (div_w + (1 if mod_w > 0 else 0)) - W
        x = F.pad(x, (0, pad_W, 0, pad_H))
        assert (
            x.shape[2] % self.patch_size[0] == 0
            and x.shape[3] % self.patch_size[1] == 0
        )
        proj_x = self.proj(x).flatten(2).transpose(1, 2)
        return (
            proj_x,
            {"height": x.shape[2], "width": x.shape[3]},
            (x.shape[2] != self.img_size[0] or x.shape[3] != self.img_size[1]),
        )


class HybridEmbed(nn.Module):
    """CNN Feature Map Embedding
    Extract feature map from CNN, flatten, project to embedding dim.
    """

    def __init__(
        self,
        backbone,
        img_size: Tuple[int],
        patch_size=Union[List, int],
        feature_size=None,
        in_chans=3,
        embed_dim=768,
    ):
        super().__init__()
        assert isinstance(backbone, nn.Module)
        if isinstance(patch_size, int):
            patch_size = to_2tuple(patch_size)
        else:
            patch_size = tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.backbone = backbone
        if feature_size is None:
            with torch.no_grad():
                # NOTE Most reliable way of determining output dims is to run forward pass
                training = backbone.training
                if training:
                    backbone.eval()
                o = self.backbone(torch.zeros(1, in_chans, img_size[0], img_size[1]))
                if isinstance(o, (list, tuple)):
                    o = o[-1]  # last feature if backbone outputs list/tuple of features
                feature_size = o.shape[-2:]
                feature_dim = o.shape[1]
                backbone.train(training)
        else:
            feature_size = to_2tuple(feature_size)
            if hasattr(self.backbone, "feature_info"):
                feature_dim = self.backbone.feature_info.channels()[-1]
            else:
                feature_dim = self.backbone.num_features

        assert feature_size[0] >= patch_size[0] and feature_size[1] >= patch_size[1]

        div_h, mod_h = divmod(feature_size[0], patch_size[0])
        div_w, mod_w = divmod(feature_size[1], patch_size[1])

        self.feature_size = (
            patch_size[0] * (div_h + (1 if mod_h > 0 else 0)),
            patch_size[1] * (div_w + (1 if mod_w > 0 else 0)),
        )
        assert (
            self.feature_size[0] % patch_size[0] == 0
            and self.feature_size[1] % patch_size[1] == 0
        )
        self.grid_size = (
            self.feature_size[0] // patch_size[0],
            self.feature_size[1] // patch_size[1],
        )
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.proj = nn.Conv2d(
            feature_dim, embed_dim, kernel_size=patch_size, stride=patch_size
        )

    def forward(self, x):
        origin_size = x.shape[-2:]
        x = self.backbone(x)
        f_h, f_w = x.shape[2:]
        # assert f_h >= self.patch_size[0] and f_w >= self.patch_size[1]

        div_h, mod_h = divmod(f_h, self.patch_size[0])
        div_w, mod_w = divmod(f_w, self.patch_size[1])

        pad_H = self.patch_size[0] * (div_h + (1 if mod_h > 0 else 0)) - f_h
        pad_W = self.patch_size[1] * (div_w + (1 if mod_w > 0 else 0)) - f_w
        x = F.pad(x, (0, pad_W, 0, pad_H))

        assert (
            x.shape[2] % self.patch_size[0] == 0
            and x.shape[3] % self.patch_size[1] == 0
        )
        if isinstance(x, (list, tuple)):
            x = x[-1]  # last feature if backbone outputs list/tuple of features

        proj_x = self.proj(x).flatten(2).transpose(1, 2)
        return (
            proj_x,
            (pad_W, pad_H),
            {"height": x.shape[2], "width": x.shape[3]},
            (x.shape[2] != self.feature_size[0] or x.shape[3] != self.feature_size[1]),
        )


class HybridEmbed1D(nn.Module):
    """CNN Feature Map Embedding which using 1D embed patching
    from https://arxiv.org/pdf/2111.08314.pdf, which benefits for text recognition task.Check paper for more detail
    Extract feature map from CNN, flatten, project to embedding dim.
    """

    def __init__(
        self,
        backbone,
        img_size: Tuple[int],
        feature_size=None,
        patch_size=1,
        in_chans=3,
        embed_dim=768,
    ):
        super().__init__()
        assert isinstance(backbone, nn.Module)
        self.img_size = img_size
        self.backbone = backbone
        self.embed_dim = embed_dim
        if feature_size is None:
            with torch.no_grad():
                # NOTE Most reliable way of determining output dims is to run forward pass
                training = backbone.training
                if training:
                    backbone.eval()
                o = self.backbone(torch.zeros(1, in_chans, img_size[0], img_size[1]))
                if isinstance(o, (list, tuple)):
                    o = o[-1]  # last feature if backbone outputs list/tuple of features
                feature_size = o.shape[-2:]
                feature_dim = o.shape[1]
                backbone.train(training)
        else:
            feature_size = to_2tuple(feature_size)
            if hasattr(self.backbone, "feature_info"):
                feature_dim = self.backbone.feature_info.channels()[-1]
            else:
                feature_dim = self.backbone.num_features

        self.window_width = patch_size
        assert feature_size[1] >= self.window_width
        div_w, mod_w = divmod(feature_size[1], self.window_width)
        self.feature_size = (
            feature_size[0],
            self.window_width * (div_w + (1 if mod_w > 0 else 0)),
        )
        assert self.feature_size[1] % self.window_width == 0
        self.grid_size = (1, self.feature_size[1] // self.window_width)
        self.num_patches = self.grid_size[1]
        self.proj = nn.Conv1d(
            feature_dim,
            embed_dim,
            kernel_size=self.window_width,
            stride=self.window_width,
            bias=True,
        )

    def forward(self, x):
        batch_size = x.shape[0]
        x = self.backbone(x)
        f_h, f_w = x.shape[2:]
        assert f_w >= self.window_width

        div_w, mod_w = divmod(f_w, self.window_width)
        pad_W = self.window_width * (div_w + (1 if mod_w > 0 else 0)) - f_w

        x = F.pad(x, (0, pad_W))
        assert x.shape[3] % self.window_width == 0

        if isinstance(x, (list, tuple)):
            x = x[-1]  # last feature if backbone outputs list/tuple of features

        proj_x = torch.zeros(
            batch_size,
            self.embed_dim,
            f_h,
            x.shape[3] // self.window_width,
            device=x.device,
            dtype=x.dtype,
        )

        for i in range(f_h):
            proj = self.proj(x[:, :, i, :])
            proj_x[:, :, i, :] = proj

        proj_x = proj_x.mean(dim=2).transpose(1, 2)  # BCHW->BCW

        return (
            proj_x,
            (pad_W,),
            {"height": x.shape[2], "width": x.shape[3]},
            (x.shape[2] != self.feature_size[0] or x.shape[3] != self.feature_size[1]),
        )

import math

import numpy as np
import torch
import torch.nn as nn
from typing import Tuple
from torch import Tensor
from torch.nn import functional as F


class Adaptive2DPositionalEncoding(nn.Module):
    """Implement Adaptive 2D positional encoder for SATRN, see
      `SATRN <https://arxiv.org/abs/1910.04396>`_
      Modified from https://github.com/Media-Smart/vedastr
      Licensed under the Apache License, Version 2.0 (the "License");
    Args:
        d_hid (int): Dimensions of hidden layer.
        n_height (int): Max height of the 2D feature output.
        n_width (int): Max width of the 2D feature output.
        dropout (int): Size of hidden layers of the model.
    """

    def __init__(
        self,
        d_hid=512,
        n_height=100,
        n_width=100,
        dropout=0.1,
    ):
        super().__init__()

        h_position_encoder = self._get_sinusoid_encoding_table(n_height, d_hid)
        h_position_encoder = h_position_encoder.transpose(0, 1)
        h_position_encoder = h_position_encoder.view(1, d_hid, n_height, 1)

        w_position_encoder = self._get_sinusoid_encoding_table(n_width, d_hid)
        w_position_encoder = w_position_encoder.transpose(0, 1)
        w_position_encoder = w_position_encoder.view(1, d_hid, 1, n_width)

        self.register_buffer("h_position_encoder", h_position_encoder)
        self.register_buffer("w_position_encoder", w_position_encoder)

        self.h_scale = self.scale_factor_generate(d_hid)
        self.w_scale = self.scale_factor_generate(d_hid)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(p=dropout)

    def _get_sinusoid_encoding_table(self, n_position, d_hid):
        """Sinusoid position encoding table."""
        denominator = torch.Tensor(
            [1.0 / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]
        )
        denominator = denominator.view(1, -1)
        pos_tensor = torch.arange(n_position).unsqueeze(-1).float()
        sinusoid_table = pos_tensor * denominator
        sinusoid_table[:, 0::2] = torch.sin(sinusoid_table[:, 0::2])
        sinusoid_table[:, 1::2] = torch.cos(sinusoid_table[:, 1::2])

        return sinusoid_table

    def scale_factor_generate(self, d_hid):
        scale_factor = nn.Sequential(
            nn.Conv2d(d_hid, d_hid, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(d_hid, d_hid, kernel_size=1),
            nn.Sigmoid(),
        )

        return scale_factor

    def init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain("ReLU"))

    def forward(self, x):
        b, c, h, w = x.size()

        avg_pool = self.pool(x)

        h_pos_encoding = self.h_scale(avg_pool) * self.h_position_encoder[:, :, :h, :]
        w_pos_encoding = self.w_scale(avg_pool) * self.w_position_encoder[:, :, :, :w]

        out = x + h_pos_encoding + w_pos_encoding

        out = self.dropout(out)

        return out


class PositionalEncoding2D(nn.Module):
    """2-D positional encodings for the feature maps produced by the encoder.
    Following https://arxiv.org/abs/2103.06450 by Sumeet Singh.
    Reference:
    https://github.com/full-stack-deep-learning/fsdl-text-recognizer-2021-labs/blob/main/lab9/text_recognizer/models/transformer_util.py
    """

    def __init__(self, d_model: int, max_h: int = 2000, max_w: int = 2000) -> None:
        super().__init__()
        self.d_model = d_model
        assert d_model % 2 == 0, f"Embedding depth {d_model} is not even"
        pe = self.make_pe(d_model, max_h, max_w)  # (d_model, max_h, max_w)
        self.register_buffer("pe", pe)

    @staticmethod
    def make_pe(d_model: int, max_h: int, max_w: int) -> Tensor:
        """Compute positional encoding."""
        pe_h = PositionalEncoding1D.make_pe(
            d_model=d_model // 2, max_len=max_h
        )  # (max_h, 1 d_model // 2)
        pe_h = pe_h.permute(2, 0, 1).expand(
            -1, -1, max_w
        )  # (d_model // 2, max_h, max_w)

        pe_w = PositionalEncoding1D.make_pe(
            d_model=d_model // 2, max_len=max_w
        )  # (max_w, 1, d_model // 2)
        pe_w = pe_w.permute(2, 1, 0).expand(
            -1, max_h, -1
        )  # (d_model // 2, max_h, max_w)

        pe = torch.cat([pe_h, pe_w], dim=0)  # (d_model, max_h, max_w)
        return pe

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass.
        Args:
            x: (B, d_model, H, W)
        Returns:
            (B, d_model, H, W)
        """
        assert x.shape[1] == self.pe.shape[0]  # type: ignore
        x = x + self.pe[:, : x.size(2), : x.size(3)]  # type: ignore
        return x


class PositionalEncoding1D(nn.Module):
    """Classic Attention-is-all-you-need positional encoding."""

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 1000) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = self.make_pe(d_model, max_len)  # (max_len, 1, d_model)
        self.register_buffer("pe", pe)

    @staticmethod
    def make_pe(d_model: int, max_len: int) -> Tensor:
        """Compute positional encoding."""
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(1)
        return pe

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass.
        Args:
            x: (S, B, d_model)
        Returns:
            (S, B, d_model)
        """
        assert x.shape[2] == self.pe.shape[2]  # type: ignore
        x = x + self.pe[: x.size(0)]  # type: ignore
        return self.dropout(x)


Size_ = Tuple[int, int]


class PosConv(nn.Module):
    # PEG  from https://arxiv.org/abs/2102.10882
    def __init__(self, in_chans, embed_dim=768, stride=1):
        super(PosConv, self).__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(in_chans, embed_dim, 3, stride, 1, bias=True, groups=embed_dim),
        )
        self.stride = stride

    def forward(self, x, size: Size_):
        B, N, C = x.shape
        cls_token, feat_token = x[:, 0], x[:, 1:]
        cnn_feat_token = feat_token.transpose(1, 2).view(B, C, *size)
        x = self.proj(cnn_feat_token)
        if self.stride == 1:
            x += cnn_feat_token
        x = x.flatten(2).transpose(1, 2)
        x = torch.cat((cls_token.unsqueeze(1), x), dim=1)
        return x

    def no_weight_decay(self):
        return ["proj.%d.weight" % i for i in range(4)]


class PosConv1D(nn.Module):
    # PEG  from https://arxiv.org/abs/2102.10882
    def __init__(self, in_chans, embed_dim=768, stride=1):
        super(PosConv1D, self).__init__()
        self.proj = nn.Sequential(
            nn.Conv1d(in_chans, embed_dim, 3, stride, 1, bias=True, groups=embed_dim),
        )
        self.stride = stride

    def forward(self, x, size: int):
        B, N, C = x.shape
        cls_token, feat_token = x[:, 0], x[:, 1:]
        cnn_feat_token = feat_token.transpose(1, 2).view(B, C, size)
        x = self.proj(cnn_feat_token)
        if self.stride == 1:
            x += cnn_feat_token
        x = x.transpose(1, 2)
        x = torch.cat((cls_token.unsqueeze(1), x), dim=1)
        return x

    def no_weight_decay(self):
        return ["proj.%d.weight" % i for i in range(4)]


def resize_pos_embed(posemb, posemb_new, num_tokens=1, gs_new=(), old_grid_shape=()):
    # Rescale the grid of position embeddings when loading from state_dict. Adapted from
    # https://github.com/google-research/vision_transformer/blob/00883dd691c63a6830751563748663526e811cee/vit_jax/checkpoint.py#L224

    print("Resized position embedding: %s to %s" % (posemb.shape, posemb_new.shape))
    ntok_new = posemb_new.shape[1]

    if num_tokens:
        posemb_tok, posemb_grid = posemb[:, :num_tokens], posemb[0, num_tokens:]
        ntok_new -= num_tokens
    else:
        posemb_tok, posemb_grid = posemb[:, :0], posemb[0]

    if not len(gs_new):  # backwards compatibility
        gs_new = [int(math.sqrt(ntok_new))] * 2

    assert len(gs_new) >= 2

    print("Position embedding grid-size from %s to %s" % (old_grid_shape, gs_new))
    posemb_grid = posemb_grid.reshape(
        1, old_grid_shape[0], old_grid_shape[1], -1
    ).permute(0, 3, 1, 2)
    posemb_grid = F.interpolate(
        posemb_grid, size=gs_new, mode="bicubic", align_corners=False
    )
    posemb_grid = posemb_grid.permute(0, 2, 3, 1).reshape(1, gs_new[0] * gs_new[1], -1)
    posemb = torch.cat([posemb_tok, posemb_grid], dim=1)

    return posemb

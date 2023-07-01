import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
import warnings

__all__ = ["ConvMLP", "ConvModule"]


class LayerNorm2d(nn.LayerNorm):
    """LayerNorm for channels of '2D' spatial BCHW tensors"""

    def __init__(self, num_channels):
        super().__init__(num_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(
            x.permute(0, 2, 3, 1),
            self.normalized_shape,
            self.weight,
            self.bias,
            self.eps,
        ).permute(0, 3, 1, 2)


class DepthwiseSeparableConv2d(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
    ):
        super(DepthwiseSeparableConv2d, self).__init__()
        self.depthwise = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=padding,
            stride=stride,
            bias=bias,
            groups=in_channels,
        )
        self.pointwise = nn.Conv2d(
            in_channels, out_channels, kernel_size=1, groups=groups, bias=bias
        )

    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out


class ConvMLP(nn.Module):
    def __init__(self, in_channels, out_channels=None, hidden_channels=None, drop=0.25):
        super().__init__()
        out_channels = in_channels or out_channels
        hidden_channels = in_channels or hidden_channels
        self.fc1 = nn.Conv2d(in_channels, hidden_channels, kernel_size=1, bias=True)
        self.norm = LayerNorm2d(hidden_channels)
        self.fc2 = nn.Conv2d(hidden_channels, out_channels, kernel_size=1, bias=True)
        self.act = nn.ReLU()
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.norm(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        return x


class ConvModule(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias="auto",
        conv_layer: Optional[nn.Module] = nn.Conv2d,
        norm_layer: Optional[nn.Module] = nn.BatchNorm2d,
        act_layer: Optional[nn.Module] = nn.ReLU,
        inplace=True,
        with_spectral_norm=False,
        padding_mode="zeros",
        order=("conv", "norm", "act"),
    ):
        official_padding_mode = ["zeros", "circular"]
        nonofficial_padding_mode = dict(
            zero=nn.ZeroPad2d, reflect=nn.ReflectionPad2d, replicate=nn.ReplicationPad2d
        )
        self.with_spectral_norm = with_spectral_norm
        self.with_explicit_padding = padding_mode not in official_padding_mode
        self.order = order
        assert isinstance(self.order, tuple) and len(self.order) == 3
        assert set(order) == set(["conv", "norm", "act"])

        self.with_norm = norm_layer is not None
        self.with_act = act_layer is not None

        if bias == "auto":
            bias = not self.with_norm
        self.with_bias = bias

        if self.with_norm and self.with_bias:
            warnings.warn("ConvModule has norm and bias at the same time")

        if self.with_explicit_padding:
            assert padding_mode in list(
                nonofficial_padding_mode
            ), "Not implemented padding algorithm"
            self.padding_layer = nonofficial_padding_mode[padding_mode]

        # reset padding to 0 for conv module
        conv_padding = 0 if self.with_explicit_padding else padding

        self.conv = conv_layer(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=conv_padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )
        self.in_channels = self.conv.in_channels
        self.out_channels = self.conv.out_channels
        self.kernel_size = self.conv.kernel_size
        self.stride = self.conv.stride
        self.padding = padding
        self.dilation = self.conv.dilation
        self.transposed = self.conv.transposed
        self.output_padding = self.conv.output_padding
        self.groups = self.conv.groups

        if self.with_spectral_norm:
            self.conv = nn.utils.spectral_norm(self.conv)

        # build normalization layers
        if self.with_norm:
            # norm layer is after conv layer
            if order.index("norm") > order.index("conv"):
                norm_channels = out_channels
            else:
                norm_channels = in_channels
            self.norm = norm_layer(norm_channels)

        if self.with_act:
            if act_layer not in [nn.Tanh, nn.PReLU, nn.Sigmoid]:
                self.act = act_layer()
            else:
                self.act = act_layer(inplace=inplace)

    def forward(self, x, activate=True, norm=True):
        for layer in self.order:
            if layer == "conv":
                if self.with_explicit_padding:
                    x = self.padding_layer(x)
                x = self.conv(x)
            elif layer == "norm" and norm and self.with_norm:
                x = self.norm(x)
            elif layer == "act" and activate and self.with_act:
                x = self.act(x)
        return x

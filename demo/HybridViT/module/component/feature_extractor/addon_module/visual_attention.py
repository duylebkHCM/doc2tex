import torch.nn as nn
import torch
from torch.nn import functional as F
from functools import reduce

__all__ = ["Adaptive_Global_Model", "GlobalContext", "SELayer", "SKBlock", "CBAM"]


def constant_init(module, val, bias=0):
    if hasattr(module, "weight") and module.weight is not None:
        nn.init.constant_(module.weight, val)
    if hasattr(module, "bias") and module.bias is not None:
        nn.init.constant_(module.bias, bias)


def xavier_init(module, gain=1, bias=0, distribution="normal"):
    assert distribution in ["uniform", "normal"]
    if distribution == "uniform":
        nn.init.xavier_uniform_(module.weight, gain=gain)
    else:
        nn.init.xavier_normal_(module.weight, gain=gain)
    if hasattr(module, "bias") and module.bias is not None:
        nn.init.constant_(module.bias, bias)


def normal_init(module, mean=0, std=1, bias=0):
    nn.init.normal_(module.weight, mean, std)
    if hasattr(module, "bias") and module.bias is not None:
        nn.init.constant_(module.bias, bias)


def uniform_init(module, a=0, b=1, bias=0):
    nn.init.uniform_(module.weight, a, b)
    if hasattr(module, "bias") and module.bias is not None:
        nn.init.constant_(module.bias, bias)


def kaiming_init(
    module, a=0, mode="fan_out", nonlinearity="relu", bias=0, distribution="normal"
):
    assert distribution in ["uniform", "normal"]
    if distribution == "uniform":
        nn.init.kaiming_uniform_(
            module.weight, a=a, mode=mode, nonlinearity=nonlinearity
        )
    else:
        nn.init.kaiming_normal_(
            module.weight, a=a, mode=mode, nonlinearity=nonlinearity
        )
    if hasattr(module, "bias") and module.bias is not None:
        nn.init.constant_(module.bias, bias)


def last_zero_init(m):
    if isinstance(m, nn.Sequential):
        constant_init(m[-1], val=0)
    else:
        constant_init(m, val=0)


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16, dropout=0.1):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(channel // reduction, channel),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x + x * y.expand_as(x)  # SE-Residual


class BasicConv(nn.Module):
    def __init__(
        self,
        in_planes,
        out_planes,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        relu=True,
        bn=True,
        bias=False,
    ):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(
            in_planes,
            out_planes,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )
        self.bn = (
            nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True)
            if bn
            else None
        )
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class ChannelGate(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=["avg", "max"]):
        super(ChannelGate, self).__init__()
        self.gate_channels = gate_channels
        self.mlp = nn.Sequential(
            Flatten(),
            nn.Linear(gate_channels, gate_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(gate_channels // reduction_ratio, gate_channels),
        )
        self.pool_types = pool_types
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.maxpool = nn.AdaptiveMaxPool2d(1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        channel_att_sum = None
        for pool_type in self.pool_types:
            if pool_type == "avg":
                avg_pool = self.avgpool(x)
                channel_att_raw = self.mlp(avg_pool)
            elif pool_type == "max":
                max_pool = self.maxpool(x)
                channel_att_raw = self.mlp(max_pool)

            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum = channel_att_sum + channel_att_raw

        scale = self.sigmoid(channel_att_sum).unsqueeze(2).unsqueeze(3).expand_as(x)
        return x * scale


class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat(
            (torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1
        )


class SpatialGate(nn.Module):
    def __init__(self):
        super(SpatialGate, self).__init__()
        kernel_size = 7
        self.compress = ChannelPool()
        self.spatial = BasicConv(
            2, 1, kernel_size, stride=1, padding=(kernel_size - 1) // 2, relu=False
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = self.sigmoid(x_out)  # broadcasting
        return x * scale


class CBAM(nn.Module):
    def __init__(
        self,
        gate_channels,
        reduction_ratio=16,
        pool_types=["avg", "max"],
        no_spatial=False,
    ):
        super(CBAM, self).__init__()
        self.ChannelGate = ChannelGate(gate_channels, reduction_ratio, pool_types)
        self.no_spatial = no_spatial
        if not no_spatial:
            self.SpatialGate = SpatialGate()

    def forward(self, x):
        x_out = self.ChannelGate(x)
        if not self.no_spatial:
            x_out = self.SpatialGate(x_out)
        return x_out


class SKBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, M=2, r=16, L=32):
        super(SKBlock, self).__init__()
        d = max(in_channels // r, L)
        self.M = M
        self.out_channels = out_channels
        self.conv = nn.ModuleList()
        for i in range(M):
            self.conv.append(
                nn.Sequential(
                    nn.Conv2d(
                        in_channels,
                        out_channels,
                        3,
                        stride,
                        padding=1 + i,
                        dilation=1 + i,
                        groups=32,
                        bias=False,
                    ),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True),
                )
            )
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Sequential(
            nn.Conv2d(out_channels, d, 1, bias=False),
            nn.BatchNorm2d(d),
            nn.ReLU(inplace=True),
        )
        self.fc2 = nn.Conv2d(d, out_channels * M, 1, 1, bias=False)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input):
        batch_size = input.size(0)
        output = []
        # the part of split
        for i, conv in enumerate(self.conv):
            # print(i,conv(input).size())
            output.append(conv(input))
        # the part of fusion
        U = reduce(lambda x, y: x + y, output)
        s = self.global_pool(U)
        z = self.fc1(s)
        a_b = self.fc2(z)
        a_b = a_b.reshape(batch_size, self.M, self.out_channels, -1)
        a_b = self.softmax(a_b)
        # the part of selection
        a_b = list(a_b.chunk(self.M, dim=1))  # split to a and b
        a_b = list(map(lambda x: x.reshape(batch_size, self.out_channels, 1, 1), a_b))
        V = list(map(lambda x, y: x * y, output, a_b))
        V = reduce(lambda x, y: x + y, V)
        return V


def make_divisible(v, divisor=8, min_value=None, round_limit=0.9):
    min_value = min_value or divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < round_limit * v:
        new_v += divisor
    return new_v


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


class GlobalContext(nn.Module):
    def __init__(
        self,
        channel,
        use_attn=True,
        fuse_add=True,
        fuse_scale=False,
        rd_ratio=1.0 / 8,
        rd_channels=None,
    ):
        super().__init__()
        self.use_attn = use_attn
        self.global_cxt = (
            nn.Conv2d(channel, 1, kernel_size=1, bias=True)
            if use_attn
            else nn.AdaptiveAvgPool2d(1)
        )

        if rd_channels is None:
            rd_channels = make_divisible(channel * rd_ratio, divisor=1, round_limit=0.0)

        if fuse_add:
            self.bottleneck_add = ConvMLP(channel, hidden_channels=rd_channels)
        else:
            self.bottleneck_add = None
        if fuse_scale:
            self.bottleneck_mul = ConvMLP(channel, hidden_channels=rd_channels)
        else:
            self.bottleneck_mul = None

        self.init_weight()

    def init_weight(self):
        if self.use_attn:
            nn.init.kaiming_normal_(
                self.global_cxt.weight, mode="fan_in", nonlinearity="relu"
            )
        if self.bottleneck_add is not None:
            nn.init.zeros_(self.bottleneck_add.fc2.weight)
        if self.bottleneck_mul is not None:
            nn.init.zeros_(self.bottleneck_mul.fc2.weight)

    def forward(self, x):
        B, C, H, W = x.shape
        if self.use_attn:
            attn = self.global_cxt(x).reshape(B, 1, H * W).squeeze(1)
            attn = F.softmax(attn, dim=-1).unsqueeze(-1)  # shape BxH*Wx1
            query = x.reshape(B, C, H * W)  # shape BxCxH*W
            glob_cxt = torch.bmm(query, attn).unsqueeze(-1)
        else:
            glob_cxt = self.global_cxt(x)
        assert len(glob_cxt.shape) == 4

        if self.bottleneck_add is not None:
            x_trans = self.bottleneck_add(glob_cxt)
            x_fuse = x + x_trans
        if self.bottleneck_mul is not None:
            x_trans = F.sigmoid(self.bottleneck_mul(glob_cxt))
            x_fuse = x * x_trans

        return x_fuse


class Adaptive_Global_Model(nn.Module):
    def __init__(self, inplanes, factor=2, ratio=0.0625, dropout=0.1):
        super(Adaptive_Global_Model, self).__init__()
        # b, w, h, c => gc_block (b, w, h, c) => => b, w, inplanes
        self.embedding = nn.Linear(inplanes * factor, inplanes)
        self.gc_block = GlobalContext(inplanes, ratio=ratio)  #
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.gc_block(x)  # BCHW => BCHW
        x = x.permute(0, 3, 1, 2)  # BCHW => BWCH
        b, w, _, _ = x.shape
        x = x.contiguous().view(b, w, -1)
        x = self.embedding(x)  # B W C
        x = self.dropout(x)
        return x

import torch.nn as nn
import torch
from torch.nn import functional as F

__all__ = ["GlobalContext"]


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

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["ASPP"]


class ASPPModule(nn.Module):
    def __init__(self, inplanes, planes, kernel_size, padding, dilation):
        super(ASPPModule, self).__init__()
        self.atrous_conv = nn.Conv2d(
            inplanes,
            planes,
            kernel_size=kernel_size,
            stride=1,
            padding=padding,
            dilation=dilation,
            bias=False,
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):  # skipcq: PYL-W0221
        x = self.atrous_conv(x)
        x = self.relu(x)
        return x


class ASPP(nn.Module):
    def __init__(
        self, inplanes: int, output_stride: int, output_features: int, dropout=0.5
    ):
        super(ASPP, self).__init__()

        if output_stride == 32:
            dilations = [1, 3, 6, 9]
        elif output_stride == 16:
            dilations = [1, 6, 12, 18]
        elif output_stride == 8:
            dilations = [1, 12, 24, 36]
        else:
            raise NotImplementedError

        self.aspp1 = ASPPModule(
            inplanes, output_features, 1, padding=0, dilation=dilations[0]
        )
        self.aspp2 = ASPPModule(
            inplanes, output_features, 3, padding=dilations[1], dilation=dilations[1]
        )
        self.aspp3 = ASPPModule(
            inplanes, output_features, 3, padding=dilations[2], dilation=dilations[2]
        )
        self.aspp4 = ASPPModule(
            inplanes, output_features, 3, padding=dilations[3], dilation=dilations[3]
        )

        self.global_avg_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(inplanes, output_features, 1, stride=1, bias=False),
            nn.ReLU(inplace=True),
        )
        self.conv1 = nn.Conv2d(output_features * 5, output_features, 1, bias=False)
        self.relu1 = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):  # skipcq: PYL-W0221
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        x5 = self.global_avg_pool(x)
        x5 = F.interpolate(x5, size=x4.size()[2:], mode="bilinear", align_corners=False)
        x = torch.cat((x1, x2, x3, x4, x5), dim=1)

        x = self.conv1(x)
        x = self.relu1(x)

        return self.dropout(x)

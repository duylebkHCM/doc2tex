from typing import Dict
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..addon_module.visual_attention import GlobalContext
from .....helper import clean_state_dict


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = self._conv3x3(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = self._conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def _conv3x3(self, in_planes, out_planes, stride=1):
        "3x3 convolution with padding"
        return nn.Conv2d(
            in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False
        )

    def zero_init_last_bn(self):
        nn.init.zeros_(self.bn2.weight)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(
        self,
        input_channel,
        output_channel,
        block,
        layers,
        with_gcb=True,
        debug=False,
        zero_init_last_bn=False,
    ):
        super(ResNet, self).__init__()
        self.with_gcb = with_gcb

        self.output_channel_block = [
            int(output_channel / 4),
            int(output_channel / 2),
            output_channel,
            output_channel,
        ]
        self.inplanes = int(output_channel / 8)

        self.conv0_1 = nn.Conv2d(
            input_channel,
            int(output_channel / 16),
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn0_1 = nn.BatchNorm2d(int(output_channel / 16))

        self.conv0_2 = nn.Conv2d(
            int(output_channel / 16),
            self.inplanes,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn0_2 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)

        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.layer1 = self._make_layer(block, self.output_channel_block[0], layers[0])
        self.conv1 = nn.Conv2d(
            self.output_channel_block[0],
            self.output_channel_block[0],
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(self.output_channel_block[0])

        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.layer2 = self._make_layer(
            block, self.output_channel_block[1], layers[1], stride=1
        )
        self.conv2 = nn.Conv2d(
            self.output_channel_block[1],
            self.output_channel_block[1],
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(self.output_channel_block[1])

        self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=(2, 1), padding=(0, 1))
        self.layer3 = self._make_layer(
            block, self.output_channel_block[2], layers[2], stride=1
        )
        self.conv3 = nn.Conv2d(
            self.output_channel_block[2],
            self.output_channel_block[2],
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn3 = nn.BatchNorm2d(self.output_channel_block[2])

        self.layer4 = self._make_layer(
            block, self.output_channel_block[3], layers[3], stride=1
        )

        self.conv4_1 = nn.Conv2d(
            self.output_channel_block[3],
            self.output_channel_block[3],
            kernel_size=2,
            stride=(2, 1),
            padding=(0, 1),
            bias=False,
        )
        self.bn4_1 = nn.BatchNorm2d(self.output_channel_block[3])

        self.conv4_2 = nn.Conv2d(
            self.output_channel_block[3],
            self.output_channel_block[3],
            kernel_size=2,
            stride=1,
            padding=0,
            bias=False,
        )
        self.bn4_2 = nn.BatchNorm2d(self.output_channel_block[3])

        self.init_weights(zero_init_last_bn=zero_init_last_bn)
        self.debug = debug

    def zero_init_last_bn(self):
        nn.init.zeros_(self.bn4_2.weight)

    def init_weights(self, zero_init_last_bn=True):
        initialized = ["global_cxt", "bottleneck_add", "bottleneck_mul"]
        for n, m in self.named_modules():
            if any([d in n for d in initialized]):
                continue
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
        if zero_init_last_bn:
            for m in self.modules():
                if hasattr(m, "zero_init_last_bn"):
                    m.zero_init_last_bn()

    def _make_layer(self, block, planes, blocks, with_gcb=False, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion

        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        if self.with_gcb:
            layers.append(GlobalContext(planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        if self.debug:
            print("input shape", x.shape)

        x = self.conv0_1(x)
        x = self.bn0_1(x)
        x = self.relu(x)

        if self.debug:
            print("conv1 shape", x.shape)

        x = self.conv0_2(x)
        x = self.bn0_2(x)
        x = self.relu(x)

        if self.debug:
            print("conv2 shape", x.shape)

        x = self.maxpool1(x)

        if self.debug:
            print("pool1 shape", x.shape)

        x = self.layer1(x)

        if self.debug:
            print("block1 shape", x.shape)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        if self.debug:
            print("conv3 shape", x.shape)

        x = self.maxpool2(x)

        if self.debug:
            print("pool2 shape", x.shape)

        x = self.layer2(x)

        if self.debug:
            print("block2 shape", x.shape)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        if self.debug:
            print("conv4 shape", x.shape)

        x = self.maxpool3(x)

        if self.debug:
            print("pool3 shape", x.shape)

        x = self.layer3(x)

        if self.debug:
            print("block3 shape", x.shape)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)

        if self.debug:
            print("conv5 shape", x.shape)

        x = self.layer4(x)

        if self.debug:
            print("block4 shape", x.shape)

        x = self.conv4_1(x)
        x = self.bn4_1(x)
        x = self.relu(x)

        if self.debug:
            print("conv6 shape", x.shape)

        x = self.conv4_2(x)
        x = self.bn4_2(x)
        x = self.relu(x)

        if self.debug:
            print("conv7 shape", x.shape)

        return x


class ResNet_FeatureExtractor(nn.Module):
    """FeatureExtractor of FAN (http://openaccess.thecvf.com/content_ICCV_2017/papers/Cheng_Focusing_Attention_Towards_ICCV_2017_paper.pdf)"""

    def __init__(
        self,
        input_channel=3,
        output_channel=512,
        gcb=False,
        pretrained=False,
        weight_dir=None,
        debug=False,
    ):
        super(ResNet_FeatureExtractor, self).__init__()
        self.ConvNet = ResNet(
            input_channel, output_channel, BasicBlock, [1, 2, 5, 3], gcb, debug
        )
        self.in_chans = input_channel
        if pretrained:
            assert weight_dir is not None
            self.load_pretrained(weight_dir)

    def forward(self, input):
        output = self.ConvNet(input)
        return output

    def load_pretrained(self, weight_dir):
        state_dict: OrderedDict = torch.load(weight_dir)
        cleaned_state_dict = clean_state_dict(state_dict)
        new_state_dict = OrderedDict()
        name: str
        param: torch.FloatTensor
        for name, param in cleaned_state_dict.items():
            if name.startswith("FeatureExtraction"):
                output_name = name.replace("FeatureExtraction.", "")
                if output_name == "ConvNet.conv0_1.weight":
                    print("Old", param.shape)
                    new_param = param.repeat(1, self.in_chans, 1, 1)
                    print("New", new_param.shape)
                else:
                    new_param = param
                new_state_dict[output_name] = new_param
        print("=> Loading pretrained weight for ResNet backbone")
        self.load_state_dict(new_state_dict)


if __name__ == "__main__":
    model = ResNet_FeatureExtractor(input_channel=1, debug=True)
    a = torch.rand(1, 1, 128, 480)
    output = model(a)
    print(output.shape)

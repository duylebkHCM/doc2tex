import torch
import torch.nn as nn

__all__ = ["vgg11_bn", "vgg13_bn", "vgg16_bn", "vgg19_bn"]


class VGG(nn.Module):
    def __init__(self, features, num_channel_out=512, init_weights=True):
        super(VGG, self).__init__()
        self.features = features
        self.num_out_features = 512

        self.lastlayer = nn.Sequential(
            nn.Conv2d(
                self.num_out_features,
                num_channel_out,
                kernel_size=1,
                stride=1,
                padding=0,
                groups=32,
                bias=False,
            ),
            nn.BatchNorm2d(num_channel_out),
            nn.ReLU(inplace=True),
        )

        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = self.lastlayer(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


def make_layers(cfg, down_sample=8, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == "M":
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        elif isinstance(v, dict):
            cur_size = v[down_sample]
            layers += [nn.MaxPool2d(kernel_size=cur_size, stride=cur_size)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


cfgs = {
    "A": [
        64,
        "M",
        128,
        "M",
        256,
        256,
        {4: (2, 1), 8: (2, 2)},
        512,
        512,
        {4: (2, 1), 8: (2, 1)},
        512,
        512,
        {4: (2, 1), 8: (2, 1)},
    ],
    "B": [
        64,
        64,
        "M",
        128,
        128,
        "M",
        256,
        256,
        {4: (2, 1), 8: (2, 2)},
        512,
        512,
        {4: (2, 1), 8: (2, 1)},
        512,
        512,
        {4: (2, 1), 8: (2, 1)},
    ],
    "D": [
        64,
        64,
        "M",
        128,
        128,
        "M",
        256,
        256,
        256,
        {4: (2, 1), 8: (2, 2)},
        512,
        512,
        512,
        {4: (2, 1), 8: (2, 1)},
        512,
        512,
        512,
        {4: (2, 1), 8: (2, 1)},
    ],
    "E": [
        64,
        64,
        "M",
        128,
        128,
        "M",
        256,
        256,
        256,
        256,
        {4: (2, 1), 8: (2, 2)},
        512,
        512,
        512,
        512,
        {4: (2, 1), 8: (2, 1)},
        512,
        512,
        512,
        512,
        {4: (2, 1), 8: (2, 1)},
    ],
}


def _vgg(
    model_path,
    cfg,
    batch_norm,
    pretrained,
    progress,
    num_channel_out,
    down_sample,
    **kwargs
):
    if pretrained:
        kwargs["init_weights"] = False
    model = VGG(
        make_layers(cfgs[cfg], down_sample, batch_norm=batch_norm),
        num_channel_out,
        **kwargs
    )
    if model_path and pretrained:
        state_dict = torch.load(model_path)
        model.load_state_dict(state_dict, strict=False)
    return model


def vgg11_bn(
    model_path="",
    num_channel_out=512,
    down_sample=8,
    pretrained=True,
    progress=True,
    **kwargs
):
    return _vgg(
        model_path,
        "A",
        True,
        pretrained,
        progress,
        num_channel_out,
        down_sample,
        **kwargs
    )


def vgg13_bn(
    model_path="",
    num_channel_out=512,
    down_sample=8,
    pretrained=True,
    progress=True,
    **kwargs
):
    return _vgg(
        model_path,
        "B",
        True,
        pretrained,
        progress,
        num_channel_out,
        down_sample,
        **kwargs
    )


def vgg16_bn(
    model_path="",
    num_channel_out=512,
    down_sample=8,
    pretrained=True,
    progress=True,
    **kwargs
):
    return _vgg(
        model_path,
        "D",
        True,
        pretrained,
        progress,
        num_channel_out,
        down_sample,
        **kwargs
    )


def vgg19_bn(
    model_path="",
    num_channel_out=512,
    down_sample=8,
    pretrained=True,
    progress=True,
    **kwargs
):
    return _vgg(
        model_path,
        "E",
        True,
        pretrained,
        progress,
        num_channel_out,
        down_sample,
        **kwargs
    )

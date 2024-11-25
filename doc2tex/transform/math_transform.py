import torch
from torch import nn
import kornia as K
import kornia.augmentation as kor
import albumentations as alb
from albumentations.pytorch.transforms import ToTensorV2
from utils.general_utils import to_3tuple


class Math_Transform(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        _mean = (
            to_3tuple(self.config["mean"])
            if self.config["rgb"]
            else self.config["mean"]
        )
        _std = (
            to_3tuple(self.config["std"]) if self.config["rgb"] else self.config["std"]
        )

        self.normalize = K.enhance.Normalize(_mean, _std)
        self.transform = kor.ImageSequential(
            kor.RandomSharpness(sharpness=(0.5, 0.5), p=0.5),
            kor.RandomBrightness(brightness=(0.5, 1.0), clip_output=True, p=0.5),
            same_on_batch=False,
            random_apply=True,
            random_apply_weights=[0.3] * 2,
        )
        self.denormalize = K.enhance.Denormalize(_mean, _std)

    @torch.no_grad()
    def forward(self, image: torch.Tensor) -> torch.Tensor:
        assert image.device.type == "cuda"
        image = torch.clamp(image, min=0.0, max=255.0)
        image = image.div(255.0)
        image = self.transform(image)
        image = self.normalize(image)
        return image


def get_test_transform(mean, std, is_gray: bool):
    test_transform = []
    normalize = [alb.Normalize(to_3tuple(mean), to_3tuple(std)), ToTensorV2()]
    if is_gray:
        test_transform += [alb.ToGray(always_apply=True)]
    test_transform += normalize

    test_transform = alb.Compose([*test_transform])
    return test_transform

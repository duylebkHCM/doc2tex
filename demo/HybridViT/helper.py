import torch
import random
import numpy as np
from PIL import Image
from typing import Dict
import torch.nn.functional as F
import numpy as np
from PIL import Image
import cv2
import math
import albumentations as alb
from albumentations.pytorch.transforms import ToTensorV2
from collections import OrderedDict
from itertools import repeat
import collections.abc


# From PyTorch internals
def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable):
            return x
        return tuple(repeat(x, n))

    return parse


to_3tuple = _ntuple(3)


def clean_state_dict(state_dict):
    # 'clean' checkpoint by removing .module prefix from state dict if it exists from parallel training
    cleaned_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] if k.startswith("module.") else k
        cleaned_state_dict[name] = v
    return cleaned_state_dict


def math_transform(mean, std, is_gray: bool):
    test_transform = []
    normalize = [
        alb.CLAHE(clip_limit=2, tile_grid_size=(2, 2), always_apply=True),
        alb.Normalize(to_3tuple(mean), to_3tuple(std)),
        ToTensorV2(),
    ]
    if is_gray:
        test_transform += [alb.ToGray(always_apply=True)]
    test_transform += normalize

    test_transform = alb.Compose([*test_transform])
    return test_transform


def pad(img: Image.Image, divable=32):
    """Pad an Image to the next full divisible value of `divable`. Also normalizes the PIL.image and invert if needed.

    Args:
        img (PIL.Image): input PIL.image
        divable (int, optional): . Defaults to 32.

    Returns:
        PIL.Image
    """
    data = np.array(img.convert("LA"))

    data = (data - data.min()) / (data.max() - data.min()) * 255
    if data[..., 0].mean() > 128:
        gray = 255 * (data[..., 0] < 128).astype(
            np.uint8
        )  # To invert the text to white
    else:
        gray = 255 * (data[..., 0] > 128).astype(np.uint8)
        data[..., 0] = 255 - data[..., 0]

    coords = cv2.findNonZero(gray)  # Find all non-zero points (text)
    a, b, w, h = cv2.boundingRect(coords)  # Find minimum spanning bounding box
    rect = data[b : b + h, a : a + w]

    if rect[..., -1].var() == 0:
        im = Image.fromarray((rect[..., 0]).astype(np.uint8)).convert("L")
    else:
        im = Image.fromarray((255 - rect[..., -1]).astype(np.uint8)).convert("L")
    dims = []

    for x in [w, h]:
        div, mod = divmod(x, divable)
        dims.append(divable * (div + (1 if mod > 0 else 0)))

    padded = Image.new("L", dims, 255)
    padded.paste(im, im.getbbox())

    return padded


def get_divisible_size(ori_h, ori_w, max_dimension=None, scale_factor=32):
    new_h, new_w = ori_h, ori_w
    if ori_h % scale_factor:
        new_h = math.ceil(ori_h / scale_factor) * scale_factor
        if new_h > max_dimension[0]:
            new_h = math.floor(ori_h / scale_factor) * scale_factor

    if ori_w % scale_factor:
        new_w = math.ceil(ori_w / scale_factor) * scale_factor
        if new_w > max_dimension[1]:
            new_w = math.floor(ori_w / scale_factor) * scale_factor

    return int(new_h), int(new_w)


def minmax_size(img, max_dimensions=None, min_dimensions=None, is_gray=True):
    if max_dimensions is not None:
        ratios = [a / b for a, b in zip(list(img.size)[::-1], max_dimensions)]
        if any([r > 1 for r in ratios]):
            size = np.array(img.size) / max(ratios)
            new_h, new_w = get_divisible_size(size[1], size[0], max_dimensions)
            img = img.resize((new_w, new_h), Image.LANCZOS)

    if min_dimensions is not None:
        ratios = [a / b for a, b in zip(list(img.size)[::-1], min_dimensions)]
        if any([r < 1 for r in ratios]):
            new_h, new_w = img.size[1] / min(ratios), img.size[0] / min(ratios)
            new_h, new_w = get_divisible_size(new_h, new_w, max_dimensions)
            if is_gray:
                MODE = "L"
                BACKGROUND = 255
            padded_im = Image.new(MODE, (new_w, new_h), BACKGROUND)
            padded_im.paste(img, img.getbbox())
            img = padded_im

    return img


def resize(resizer, img: Image.Image, opt: Dict):
    # for math recognition problem image alway in grayscale mode
    img = img.convert("L")
    assert isinstance(opt, Dict)
    assert "imgH" in opt
    assert "imgW" in opt
    expected_H = opt["imgH"]

    if expected_H is None:
        max_dimensions = opt[
            "max_dimension"
        ]  # can be bigger than max dim in training set
        min_dimensions = opt["min_dimension"]
        # equal to min dim in trainign set
        test_transform = math_transform(opt["mean"], opt["std"], not opt["rgb"])
        try:
            new_img = minmax_size(
                pad(img) if opt["pad"] else img,
                max_dimensions,
                min_dimensions,
                not opt["rgb"],
            )

            if not resizer:
                new_img = np.asarray(new_img.convert("RGB")).astype("uint8")
                new_img = test_transform(image=new_img)["image"]
                if not opt["rgb"]:
                    new_img = new_img[:1]
                new_img = new_img.unsqueeze(0)
                new_img = new_img.float()
            else:
                with torch.no_grad():
                    input_image = new_img.convert("RGB").copy()
                    r, w, h = 1, input_image.size[0], input_image.size[1]
                    for i in range(20):
                        h = int(h * r)
                        new_img = pad(
                            minmax_size(
                                input_image.resize(
                                    (w, h), Image.BILINEAR if r > 1 else Image.LANCZOS
                                ),
                                max_dimensions,
                                min_dimensions,
                                not opt["rgb"],
                            )
                        )
                        t = test_transform(
                            image=np.array(new_img.convert("RGB")).astype("uint8")
                        )["image"]
                        if not opt["rgb"]:
                            t = t[:1]
                        t = t.unsqueeze(0)
                        t = t.float()
                        w = (resizer(t.to(opt["device"])).argmax(-1).item() + 1) * opt[
                            "min_dimension"
                        ][1]

                        if w == new_img.size[0]:
                            break

                        r = w / new_img.size[0]

                new_img = t
        except ValueError as e:
            print("Error:", e)
            new_img = np.asarray(img.convert("RGB")).astype("uint8")
            assert len(new_img.shape) == 3 and new_img.shape[2] == 3
            new_img = test_transform(image=new_img)["image"]
            if not opt["rgb"]:
                new_img = new_img[:1]
            new_img = new_img.unsqueeze(0)
            h, w = new_img.shape[2:]
            new_img = F.pad(
                new_img, (0, max_dimensions[1] - w, 0, max_dimensions[0] - h), value=1
            )

    assert len(new_img.shape) == 4, f"{new_img.shape}"
    return new_img

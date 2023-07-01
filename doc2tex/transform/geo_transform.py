from torchvision.transforms import functional as FT
import numpy as np
from PIL import Image
import itertools
import random


def random_rotation(img):
    w, h = img.size
    r_angle = np.arctan(h * 1.0 / w)
    ratio = random.uniform(3.0, 4.0)
    angle = (r_angle * 180) / (np.pi * ratio)
    angle = min(angle, 2.0)
    angle = random.uniform(-angle, angle)
    img = FT.rotate(img, angle=angle, fill=255.0, interpolation=Image.BILINEAR)
    return img


def random_scale(trim_img: Image.Image, pad_img: Image.Image, scale_ratio=(0.3, 0.3)):
    trim_h, trim_w = trim_img.size[::-1]
    pad_h, pad_w = pad_img.size[::-1]

    remain_h, remain_w = (pad_h - trim_h), (pad_w - trim_w)
    scale_h, scale_w = trim_h, trim_w

    if remain_h > 0:
        randrange_height = np.linspace(0.0, scale_ratio[0], num=10)
        scale_h = random.choice(randrange_height.tolist())
        scale_h = trim_h + scale_h * remain_h
    if remain_w > 0:
        randrange_width = np.linspace(0.1, scale_ratio[0], num=10)
        scale_w = random.choice(randrange_width.tolist())
        scale_w = trim_w + scale_w * remain_w

    if scale_w != trim_w and scale_h != trim_h:
        scale_img = trim_img.resize(
            (int(scale_w), int(scale_h)), resample=Image.BILINEAR
        )
        return scale_img

    return trim_img


def geometry_transform(np_ar):
    rows = [(row == 255).all() for row in np_ar]
    cols = [(row == 255).all() for row in np_ar.transpose()]

    top = len([x for x in itertools.takewhile(lambda x: x, rows)])
    bottom = len(rows) - len([x for x in itertools.takewhile(lambda x: x, rows[::-1])])

    left = len([x for x in itertools.takewhile(lambda x: x, cols)])
    right = len(cols) - len([x for x in itertools.takewhile(lambda x: x, cols[::-1])])

    new_img = Image.new("L", (np_ar.shape[1], np_ar.shape[0]), color=255)
    trim_img = np_ar[top:bottom, left:right]
    trim_img = Image.fromarray(trim_img).convert("L")
    scale_img = random_scale(trim_img, new_img)
    trim_h, trim_w = scale_img.size[::-1]

    offset_x = (
        random.randint(0, int(np_ar.shape[1] - trim_w))
        if int(np_ar.shape[1] - trim_w) > 0
        else 0
    )
    offset_y = (
        random.randint(0, int(np_ar.shape[0] - trim_h))
        if int(np_ar.shape[0] - trim_h) > 0
        else 0
    )

    if offset_x > 0 and offset_y > 0:
        new_img.paste(trim_img, (offset_x, offset_y))
        if random.random() > 0.5:
            new_img = random_rotation(new_img)
        new_img = np.asarray(new_img).astype("uint8")
        return new_img

    return np_ar

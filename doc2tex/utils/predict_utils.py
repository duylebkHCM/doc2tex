import os
import torch
import cv2
import numpy as np
from PIL import Image
from typing import Dict
import torch.nn.functional as F
from .general_utils import to_3tuple
from torchvision import transforms
from .data_utils import minmax_size, pad
from transform.math_transform import get_test_transform as math_transform


def resize(resizer, img_path, opt: Dict):
    # for math recognition problem image alway in grayscale mode
    img = Image.open(img_path).convert("L")
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
            if opt.get("downsample", None) is not None:
                ratio = opt["downsample"]
                w, h = img.size
                if (
                    h / ratio >= opt["min_dimension"][0]
                    and w / ratio >= opt["min_dimension"][1]
                ):
                    img = np.asarray(img).astype("uint8")
                    img = cv2.resize(
                        img,
                        dsize=(int(w / ratio), int(h / ratio)),
                        interpolation=cv2.INTER_AREA,
                    )
                    img = Image.fromarray(img).convert("L")

            new_img = minmax_size(
                pad(img) if opt["pad"] else img,
                max_dimensions,
                min_dimensions,
                not opt["rgb"],
            ).convert("RGB")

            if not opt["use_resizer"] and resizer is None:
                new_img = np.asarray(new_img).astype("uint8")
                new_img = test_transform(image=new_img)["image"]
                if not opt["rgb"]:
                    new_img = new_img[:1]
                new_img = new_img.unsqueeze(0)
                new_img = new_img.float()
            else:
                with torch.no_grad():
                    input_image = pad(new_img).convert("RGB").copy()
                    r, w = 1, input_image.size[0]
                    for i in range(10):
                        new_img = minmax_size(
                            input_image.resize(
                                (w, int(input_image.size[1] * r)),
                                Image.BILINEAR if r > 1 else Image.LANCZOS,
                            ),
                            max_dimensions,
                            min_dimensions,
                        )
                        t = test_transform(image=np.array(pad(new_img).convert("RGB")))[
                            "image"
                        ].unsqueeze(0)
                        w = (resizer(t.to(opt["device"])).argmax(-1).item() + 1) * opt[
                            "min_width"
                        ]
                        if w == new_img.size[0]:
                            break
                        r = w / new_img.size[0]
                new_img = t

        except ValueError as e:
            print("Error:", e)
            new_img = np.asarray(img).astype("uint8")
            assert len(new_img.shape) == 3 and new_img.shape[2] == 3
            new_img = test_transform(image=new_img)["image"]
            if not opt["rgb"]:
                new_img = new_img[:1]
            new_img = new_img.unsqueeze(0)
            h, w = new_img.shape[2:]
            new_img = F.pad(
                new_img, (0, max_dimensions[1] - w, 0, max_dimensions[0] - h), value=1
            )
    else:
        if not opt["rgb"]:
            img = img.convert("L")
        img = np.asarray(img).astype("uint8")

        if len(img.shape) == 2:
            img = np.expand_dims(img, axis=-1)
            img = np.repeat(img, 3, axis=-1)

        # img_bw = min(
        #     img, expected_H, opt["min_dimension"][1], opt["min_dimension"][0], "None"
        # )
        img = torch.FloatTensor(img)
        img = img.permute(2, 0, 1)
        img = transforms.Normalize(to_3tuple(opt["mean"]), to_3tuple(opt["std"]))(img)
        if not opt["rgb"]:
            img = img[:1]
        new_img = img.unsqueeze(0)

    assert len(new_img.shape) == 4, f"{new_img.shape}"
    return new_img

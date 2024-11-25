from inspect import stack
from typing import Iterable, List
import numpy as np
import cv2
import os
import sys
from PIL import Image
from tqdm import tqdm
from pathlib import Path


def vstack_autopad(images: Iterable[np.array], pad_value: int = 0) -> np.ndarray:
    max_width = 0
    for img in images:
        max_width = max(max_width, img.shape[1])

    padded_images = []
    for img in images:
        width = img.shape[1]
        pad_top = 0
        pad_bottom = 0
        pad_left = 0
        pad_right = max_width - width
        img = cv2.copyMakeBorder(
            img,
            pad_top,
            pad_bottom,
            pad_left,
            pad_right,
            cv2.BORDER_CONSTANT,
            value=pad_value,
        )
        padded_images.append(img)

    return np.vstack(padded_images)


if __name__ == "__main__":
    image_pred = sys.argv[1]
    image_gold = sys.argv[2]
    output_dir = sys.argv[3]
    output_dir = Path(image_gold).parent / output_dir
    if not os.path.exists(str(output_dir)):
        os.makedirs(str(output_dir), exist_ok=True)

    for img_name in tqdm(os.listdir(image_gold)):
        if os.path.exists(os.path.join(image_pred, img_name)):
            try:
                img = Image.open(os.path.join(image_pred, img_name)).convert("RGB")
                gold = Image.open(os.path.join(image_gold, img_name)).convert("RGB")
                img = np.asarray(img, dtype=np.uint8)
                gold = np.asarray(gold, dtype=np.uint8)
                stack_img = vstack_autopad((gold, img))
                stack_img = Image.fromarray(stack_img).save(
                    os.path.join(str(output_dir), img_name)
                )
            except Exception as e:
                print(e)
                continue

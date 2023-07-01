import random
import torch
from typing import List, Tuple
import cv2
import numpy as np
from transform.geo_transform import geometry_transform


class ClusterCollate:
    def __init__(self, opt, image_padding_value=0) -> None:
        self.min_width, self.min_height = opt["min_dimension"]
        self.opt = opt
        self.image_padding_value = image_padding_value

    def collate_images(
        self, images: List[np.ndarray], new_sizes: List[Tuple[int, int]]
    ):
        new_images = []
        for img in images:
            assert img.dtype == "uint8"

            if not self.opt["rgb"] and random.random() > 0.5:
                # currently only support grayscale image a.k.a printed math expression
                img = geometry_transform(img)

            if len(img.shape) == 2:
                img = np.expand_dims(img, axis=-1)
            img = torch.from_numpy(img).float()
            img = img.permute(2, 0, 1)
            new_images.append(img)

        batch_pad_img = torch.cat([t.unsqueeze(0) for t in new_images], dim=0)

        del new_images

        return batch_pad_img

    def collate_texts(self, texts):
        return [text.strip().split() for text in texts]

    def __call__(self, batch):
        batch = filter(lambda x: x is not None, batch)
        images, labels, new_sizes, names = zip(*batch)
        image_tensors = self.collate_images(images, new_sizes)
        labels = self.collate_texts(labels)

        return image_tensors, labels, names

import PIL
from PIL import Image
import numpy as np
import subprocess
from timeit import Timer


def crop_image(img, output_path):
    old_im = Image.open(img).convert("L")
    img_data = np.asarray(old_im, dtype=np.uint8)  # height, width

    nnz_inds = np.where(img_data != 255)
    if len(nnz_inds[0]) == 0:
        # old_im.save(output_path)
        return False

    y_min = np.min(nnz_inds[0])
    y_max = np.max(nnz_inds[0])
    x_min = np.min(nnz_inds[1])
    x_max = np.max(nnz_inds[1])

    old_im = old_im.crop((x_min, y_min, x_max + 1, y_max + 1))
    old_im.save(output_path)

    return True

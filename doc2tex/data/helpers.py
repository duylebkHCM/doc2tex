import math


def get_divisible_size(ori_h, ori_w, max_dimension=None, scale_factor=32):
    if ori_h % scale_factor:
        new_h = math.ceil(ori_h / scale_factor) * scale_factor
        if max_dimension and (new_h > max_dimension[0]):
            new_h = math.floor(ori_h / scale_factor) * scale_factor
    if ori_w % scale_factor:
        new_w = math.ceil(ori_w / scale_factor) * scale_factor
        if max_dimension and (new_w > max_dimension[1]):
            new_w = math.floor(ori_w / scale_factor) * scale_factor
    return int(new_h), int(new_w)


def get_size(ori_w, ori_h, config):
    if config.get("downsample", 1) <= 1:
        return ori_h, ori_w

    ori_h, ori_w = ori_h / config["downsample"], ori_w / config["downsample"]
    min_dim, max_dim = config["min_dimension"], config["max_dimension"]

    scale_factor = config["scale_factor"]
    new_h, new_w = get_divisible_size(ori_h, ori_w, scale_factor=scale_factor)

    if any(
        [
            dim % scale_factor != 0
            for limit_size in (min_dim, max_dim)
            for dim in limit_size
        ]
    ):
        raise ValueError("Min max dimension should divisible by scale factor")

    ratios = [a / b for a, b in zip((new_h, new_w), tuple(max_dim))]
    if any([r > 1 for r in ratios]):
        new_h, new_w = new_h // max(ratios), new_w // max(ratios)
        new_h, new_w = get_divisible_size(new_h, new_w, max_dim, scale_factor)

    ratios = [a / b for a, b in zip((new_h, new_w), tuple(min_dim))]
    if any([r < 1 for r in ratios]):
        new_h, new_w = new_h // max(ratios), new_w // max(ratios)
        new_h, new_w = get_divisible_size(new_h, new_w, scale_factor)

    return new_h, new_w

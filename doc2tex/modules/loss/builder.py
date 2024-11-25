from torch import nn
from modules.loss.labelsmoothing import LabelSmoothingLoss
from typing import Dict


def criterion_kwargs(cfg: Dict):
    """cfg/argparse to kwargs helper
    Convert optimizer args in argparse args or cfg like object to keyword args for updated create fn.
    """
    kwargs = dict(ignore_index=cfg["ignore_index"], reduction=cfg["reduction"])
    if cfg.get("weight", None) is not None:
        kwargs["weight"] = cfg["weight"]
    if cfg.get("loss_args", None) is not None:
        kwargs.update(cfg["loss_args"])
    return kwargs


def create_criterion(loss, loss_kwargs):
    if loss == "smooth":
        criterion = LabelSmoothingLoss(**loss_kwargs)
    elif loss == "entropy":
        criterion = nn.CrossEntropyLoss(**loss_kwargs)

    return criterion

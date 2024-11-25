from typing import Dict, Optional, Sequence

import torch
import torch.nn as nn
import torch.optim as optim

from .adamp import AdamP
from .lamb import Lamb
from .lookahead import Lookahead
from .madgrad import MADGRAD


def add_weight_decay(model, weight_decay=1e-5, skip_list=()):
    decay = []
    no_decay = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights
        if len(param.shape) == 1 or name.endswith(".bias") or name in skip_list:
            no_decay.append(param)
        else:
            decay.append(param)
    return [
        {"params": no_decay, "weight_decay": 0.0},
        {"params": decay, "weight_decay": weight_decay},
    ]


def optimizer_kwargs(cfg: Dict):
    """cfg/argparse to kwargs helper
    Convert optimizer args in argparse args or cfg like object to keyword args for updated create fn.
    """
    kwargs = dict(
        opt=cfg["opt"],
        lr=cfg["lr"],
        weight_decay=cfg["weight_decay"],
        momentum=cfg["momentum"],
    )
    if cfg.get("opt_eps", None) is not None:
        kwargs["eps"] = cfg["opt_eps"]
    if cfg.get("opt_betas", None) is not None:
        kwargs["betas"] = cfg["opt_betas"]
    if cfg.get("opt_args", None) is not None:
        kwargs.update(cfg["opt_args"])
    return kwargs


def create_optimizer(
    model,
    opt,
    lr,
    weight_decay,
    momentum,
    filter_bias_and_bn,
    layer_decay=1.0,
    train_mode="scratch",
    **kwargs
):
    if weight_decay and filter_bias_and_bn:
        skip = {}
        if hasattr(model, "no_weight_decay"):
            skip = model.no_weight_decay()

        parameters = add_weight_decay(model, weight_decay, skip)
        weight_decay = 0.0
    else:
        parameters = model.parameters()

    opt_lower = opt.lower()
    opt_split = opt_lower.split("_")
    opt_lower = opt_split[-1]
    opt_args = dict(weight_decay=weight_decay, **kwargs)

    if lr is not None:
        opt_args.setdefault("lr", lr)
    if opt_lower == "adam":
        optimizer = optim.Adam(parameters, **opt_args)
    elif opt_lower == "adamw":
        optimizer = optim.AdamW(parameters, **opt_args)
    elif opt_lower == "adamp":
        optimizer = AdamP(parameters, wd_ratio=0.01, nesterov=True, **opt_args)
    elif opt_lower == "adadelta":
        optimizer = optim.Adadelta(parameters, **opt_args)
    elif opt_lower == "adagrad":
        opt_args.setdefault("eps", 1e-8)
        optimizer = optim.Adagrad(parameters, **opt_args)
    elif opt_lower == "lamb":
        optimizer = Lamb(parameters, **opt_args)
    elif opt_lower == "madgrad":
        optimizer = MADGRAD(parameters, momentum=momentum, **opt_args)

    if len(opt_split) > 1:
        if opt_split[0] == "lookahead":
            optimizer = Lookahead(optimizer)

    return optimizer

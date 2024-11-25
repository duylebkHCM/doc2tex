from typing import List, Union
from pathlib import Path
import yaml
import glob
import os
from collections import OrderedDict

import torch
from torch import nn
from modules import build_model
from modules.converter import builder
from collections import OrderedDict
import csv
import math

from modules.component.feature_extractor import ResNet_FeatureExtractor
from modules.component.seq_modeling.addon_module import PatchEmbed, HybridEmbed
from modules.component.common import resize_pos_embed


class Averager(object):
    """Compute average for torch.Tensor, used for loss average."""

    def __init__(self):
        self.reset()

    def add(self, v):
        if isinstance(v, torch.Tensor):
            count = v.data.numel()
            v = v.data.sum()
        elif isinstance(v, list):
            count = len(v)
        elif isinstance(v, int):
            count = 1

        self.n_count += count
        self.sum += v

    def reset(self):
        self.n_count = 0
        self.sum = 0

    def val(self):
        res = 0
        if self.n_count != 0:
            res = self.sum / float(self.n_count)
        return res


def update_summary(
    iteration, train_metrics, eval_metrics, filename, lr=None, write_header=False
):
    if iteration == 0:
        return
    rowd = OrderedDict(iteration=iteration)
    rowd.update([("train_" + k, v) for k, v in train_metrics.items()])
    rowd.update([("eval_" + k, v) for k, v in eval_metrics.items()])

    if lr is not None:
        rowd["lr"] = lr

    with open(filename, mode="a") as cf:
        dw = csv.DictWriter(cf, fieldnames=rowd.keys())
        if write_header:  # first iteration (epoch == 1 can't be used)
            dw.writeheader()
        dw.writerow(rowd)


def pytorch_count_params(model):
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params


def extract_weight_from_ckpt(paths: List[Union[str, Path]]):
    for path in paths:
        if isinstance(path, Path):
            path = str(path)
        config_name = glob.glob(os.path.join(path, "*.yaml.*"))[0]
        config_name = config_name[:-4]
        config = yaml.load(
            os.path.join(
                "../../config/train/paper_experiments/report_paper", config_name
            ),
            Loader=yaml.FullLoader,
        )
        config["vocab"] = "../" + config["vocab"]

        converter = builder.create_converter(config)
        config["num_class"] = len(converter.character)
        model = build_model.Model(config)
        model.load_state_dict(
            torch.load(os.path.join(path, "best_accuracy.pth"), map_location="cpu")[
                "model"
            ]
        )
        torch.save(model.state_dict(), os.path.join(path, "model_weight.pth"))


def adjust_learning_rate(optimizer, epoch, args):
    """Decay the learning rate with half-cycle cosine after warmup"""
    if epoch < args["warmup_epochs"]:
        lr = args["lr"] * epoch / args["warmup_epochs"]
    else:
        lr = args["min_lr"] + (args["lr"] - args["min_lr"]) * 0.5 * (
            1.0
            + math.cos(
                math.pi
                * (epoch - args["warmup_epochs"])
                / (args["epochs"] - args["warmup_epochs"])
            )
        )
    for param_group in optimizer.param_groups:
        if "lr_scale" in param_group:
            param_group["lr"] = lr * param_group["lr_scale"]
        else:
            param_group["lr"] = lr
    return lr


def save_checkpoint(
    model, configim, best_acc, best_norm_ED, best_word_ED, best_bleu, iter, save_path
):
    state = {
        "model": model.state_dict(),
        "optimizer": configim.state_dict(),
        "best_acc": best_acc,
        "best_ED": best_norm_ED,
        "best_word_ED": best_word_ED,
        "best_bleu": best_bleu,
        "iter": iter + 1,
    }

    torch.save(state, save_path)


def load_checkpoint(config, model: nn.Module, configimizer=None):
    current_device = "cuda" if next(model.parameters()).is_cuda else "cpu"

    if config.get("pretrained_weight", "") != "":
        print("Initial weight")
        checkpoint = torch.load(
            config["pretrained_weight"], map_location=current_device
        )
        assert checkpoint.get("model", None) is not None
        print("Update weight")
        model_state_dict = checkpoint["model"]
        model.load_state_dict(model_state_dict, strict=False)

    if config["saved_model"] != "":
        try:
            print(f'Load checkpoint from {config["saved_model"]} and continue training')
            checkpoint = torch.load(config["saved_model"], map_location=current_device)
            if "model" in checkpoint:
                state_dict = checkpoint["model"]
            updated_param = {}
            for name, param in state_dict.items():
                if name == "seqmodeler.SequenceModeling.pos_embed":
                    if (
                        param.shape[1]
                        != model.seqmodeler.SequenceModeling.pos_embed.shape[1]
                    ):
                        if config["SequenceModeling"]["params"]["backbone"] is not None:
                            seq_modeling = config["SequenceModeling"]["params"]
                            if seq_modeling["backbone"]["name"] == "resnet":
                                param_kwargs = dict()
                                if (
                                    seq_modeling["backbone"].get("pretrained", None)
                                    is not None
                                ):
                                    param_kwargs["pretrained"] = seq_modeling[
                                        "backbone"
                                    ]["pretrained"]
                                if (
                                    seq_modeling["backbone"].get("weight_dir", None)
                                    is not None
                                ):
                                    param_kwargs["weight_dir"] = seq_modeling[
                                        "backbone"
                                    ]["weight_dir"]
                                backbone = ResNet_FeatureExtractor(
                                    seq_modeling["backbone"]["input_channel"],
                                    seq_modeling["backbone"]["output_channel"],
                                    seq_modeling["backbone"]["gcb"],
                                    **param_kwargs,
                                )
                            old_patched = HybridEmbed(
                                backbone=backbone,
                                img_size=(
                                    config["old_max_height"],
                                    config["old_max_width"],
                                ),
                                patch_size=config["SequenceModeling"]["params"][
                                    "patch_size"
                                ],
                                in_chans=config["SequenceModeling"]["params"][
                                    "input_channel"
                                ],
                            )
                        else:
                            old_patched = PatchEmbed(
                                img_size=(
                                    config["old_max_height"],
                                    config["old_max_width"],
                                ),
                                patch_size=config["SequenceModeling"]["params"][
                                    "patch_size"
                                ],
                            )
                        old_gs = old_patched.grid_size
                        param = resize_pos_embed(
                            posemb=param,
                            posemb_new=model.seqmodeler.SequenceModeling.pos_embed,
                            old_grid_shape=old_gs,
                            gs_new=model.seqmodeler.SequenceModeling.patch_embed.grid_size,
                        )
                        updated_param[name] = param
            state_dict.update(updated_param)

            model.load_state_dict(state_dict)

            configimizer.load_state_dict(checkpoint["optimizer"])
            best_accuracy = checkpoint["best_acc"]
            best_norm_ED = checkpoint["best_ED"]
            best_word_ED = checkpoint.get("best_word_ED", 0)
            start_iter = checkpoint["iter"]
            best_bleu = checkpoint["best_bleu"]
        except Exception as e:
            print(e)
            pass
    else:
        best_accuracy = -1
        best_norm_ED = -1
        best_word_ED = -1
        best_bleu = -1
        start_iter = 0

    return best_accuracy, best_bleu, best_norm_ED, best_word_ED, start_iter

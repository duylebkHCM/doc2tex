import os
import sys
import time
import json
import random
import argparse
import yaml
import numpy as np
from pathlib import Path

import torch
import torch.utils.data
import torch.backends.cudnn as cudnn

sys.path.insert(0, str(Path(__file__).resolve().parent))
from doc2tex.engine.training import init_training, train_one_step, validation
from doc2tex.utils.model_utils import load_checkpoint, Averager
from doc2tex.data.builder import build_loader


def train(config, args, log_dir, device):
    scaler, model, configimizer, criterion, converter = init_training(
        config, args, log_dir, device
    )

    best_accuracy, best_bleu, best_norm_ED, best_word_ED, start_iter = load_checkpoint(
        config, model, configimizer
    )

    train_loader, valid_loader, augment = build_loader(config, device)

    """Create history logging metric and loss values"""
    train_loss_avg = Averager()

    """ start training """
    model.train(mode=True)

    start_time = time.time()
    iteration = start_iter

    data_iter = iter(train_loader)
    model.zero_grad()

    while True:
        # train part
        cur_lr, train_loss_avg = train_one_step(
            train_loader,
            data_iter,
            configimizer,
            criterion,
            converter,
            config,
            iteration,
            device,
            augment,
            model,
            scaler,
            config["scheduler"],
            train_loss_avg,
        )

        # validation part
        if ((iteration + 1) % config.get("accum_grad", 1) == 0) and (
            ((iteration + 1) % config["valInterval"] == 0) or iteration == 0
        ):  # To see training progress, we also conduct validation when 'iteration == 0'
            best_accuracy, best_bleu, best_norm_ED, best_word_ED = validation(
                iteration,
                model,
                configimizer,
                cur_lr,
                log_dir,
                start_time,
                augment,
                criterion,
                valid_loader,
                converter,
                config,
                args,
                device,
                train_loss_avg,
                best_accuracy,
                best_bleu,
                best_norm_ED,
                best_word_ED,
            )

        if (iteration + 1) == config["num_iter"]:
            print("end the training")
            sys.exit()

        iteration += 1

        if config["sanity_check"]:
            break


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="", help="Path to config yaml file")
    parser.add_argument(
        "--amp",
        action="store_true",
        default=False,
        help="Use mix precision to speed up training",
    )
    parser.add_argument(
        "--resume_path",
        type=str,
        default=None,
        help="Path to checkpoint to continue training",
    )
    parser.add_argument(
        "--accum-grad",
        action="store_true",
        default=False,
        help="Perform gradient accummulation",
    )
    parser.add_argument("--compile", action="store_true", default=False)
    args = parser.parse_args()

    config = yaml.load(open(args.config), Loader=yaml.FullLoader)
    config["use_amp"] = args.amp and torch.cuda.is_available()
    config["saved_model"] = "" if not args.resume_path else args.resume_path
    config["exp_name"] = args.resume_path.split("/")[-2] if args.resume_path else None

    random.seed(config["manualSeed"])
    np.random.seed(config["manualSeed"])
    torch.manual_seed(config["manualSeed"])

    if torch.cuda.is_available():
        torch.cuda.manual_seed(config["manualSeed"])
        cudnn.benchmark = False
        cudnn.deterministic = False
        config["num_gpu"] = torch.cuda.device_count()
        device = "cuda"
    else:
        config["num_gpu"] = 0
        device = "cpu"

    if config["workers"] <= 0:
        config["workers"] = (os.cpu_count() // 2) // config["num_gpu"]

    if config["num_gpu"] > 1:
        config["batch_size"] = config["num_gpu"] * config["batch_size"]

    log_dir = os.path.relpath(
        Path(args.config).absolute(),
        start=Path(__file__).parent.joinpath("config").absolute(),
    )
    log_dir = log_dir.replace(".yaml", "")
    log_dir = f"./saved_models/{log_dir}"
    if not Path(log_dir).exists():
        Path(log_dir).mkdir(parents=True)
    print("LOG DIR", log_dir)

    train(config, args, log_dir, device)

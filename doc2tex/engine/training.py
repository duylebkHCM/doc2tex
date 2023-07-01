import os
import sys
import time
import numpy as np
from copy import deepcopy
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.utils.data
from utils.model_utils import adjust_learning_rate, update_summary, save_checkpoint
from modules.build_model import Model
from modules.converter import create_converter
from modules.loss import create_criterion, criterion_kwargs
from modules.optim import create_optimizer, optimizer_kwargs
from .inferencing import validation_step


def init_training(config, args, log_dir, device):
    """Initial accellerator"""
    scaler = None
    if config["use_amp"]:
        scaler = torch.cuda.amp.GradScaler()

    """ model configuration """
    converter = create_converter(config, device)
    config["num_class"] = len(converter.character)
    model = Model(config)
    params_num = []
    for p in filter(lambda p: p.requires_grad, model.parameters()):
        params_num.append(np.prod(p.size()))
    with open(f"{log_dir}/log_train.txt", "a") as log:
        log.write(f"Trainable params num: {str(sum(params_num))}\n")

    has_compile = hasattr(torch, "compile")
    if has_compile and args.compile:
        import torch._dynamo as dynamo

        dynamo.reset()
        model = torch.compile(model, mode="reduce-overhead")
    model = model.to(device)

    """ setup loss """
    loss_args: dict = config["criterion"].get("loss_args", None)
    if loss_args is not None:
        if "classes" in list(loss_args.keys()):
            loss_args["classes"] = len(converter.character)
    loss_config = criterion_kwargs(config["criterion"])
    loss_config["ignore_index"] = converter.ignore_idx
    print(loss_config)
    criterion = create_criterion(config["criterion"]["name"], loss_config).to(device)

    """ setup optimizer """
    configimizer_config = optimizer_kwargs(config["optimizer"])
    configimizer = create_optimizer(
        model, filter_bias_and_bn=config["filter_bias_and_bn"], **configimizer_config
    )
    print("configimizer:")
    print(configimizer)

    """ final configions """
    # Save configuration information
    with open(f"{log_dir}/{args.config.split(os.sep)[-1]}.txt", "w") as config_file:
        config_log = "------------ options -------------\n"
        for k, v in config.items():
            config_log += f"{str(k)}: {str(v)}\n"
        config_log += "---------------------------------------\n"
        print(config_log)
        config_file.write(config_log)

    return scaler, model, configimizer, criterion, converter


def forward_step(converter, model, criterion, config, image, labels):
    if "Attn" in config["Prediction"]["name"]:
        text, length = converter.encode(
            labels, batch_max_length=config["batch_max_length"]
        )
        _, preds, _ = model(image, text[:, :-1])  # align with Attention.forward
        target = text[:, 1:]  # without [GO] Symbol
        cost = criterion(preds.view(-1, preds.shape[-1]), target.contiguous().view(-1))
    elif config["Prediction"]["name"] == "TFM":
        text, length = converter.encode(
            labels, batch_max_length=config["batch_max_length"]
        )
        _, preds, _ = model(image, text[:, :-1])
        target = text[:, 1:]  # without [GO] Symbol
        cost = criterion(preds.view(-1, preds.shape[-1]), target.contiguous().view(-1))
    return cost


def train_one_step(
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
    scheduler,
    train_loss_avg,
):
    param_groups = deepcopy(configimizer.param_groups)
    cur_lr = [param_group["lr"] for param_group in param_groups]
    cur_lr = sum(cur_lr) / len(cur_lr)

    try:
        image_tensors, labels, _ = next(data_iter)
    except StopIteration:
        data_iter = iter(train_loader)
        image_tensors, labels, _ = next(data_iter)

    assert image_tensors.device.type == device.type

    if config["augment"]:
        assert not config["cpu_augment"]
        image_tensors = augment(image_tensors)

    if config["use_amp"]:
        with torch.cuda.amp.autocast():
            cost = forward_step(
                converter, model, criterion, config, image_tensors, labels
            )
    else:
        cost = forward_step(converter, model, criterion, config, image_tensors, labels)

    loss = cost.mean()
    loss = loss / config.get("accum_grad", 1)

    if config["use_amp"]:
        scaler.scale(loss).backward()
        scaler.unscale_(configimizer)
        if config["grad_clip"] > 0:
            nn.utils.clip_grad_norm_(model.parameters(), config["grad_clip"])
        if (iteration + 1) % config.get("accum_grad", 1) == 0:
            scaler.step(configimizer)
            scaler.update()
            model.zero_grad()
    else:
        loss.backward()
        if config["grad_clip"] > 0:
            nn.utils.clip_grad_norm_(
                model.parameters(), config["grad_clip"]
            )  # gradient clipping with 5 (Default)
        if (iteration + 1) % config.get("accum_grad", 1) == 0:
            configimizer.step()
            model.zero_grad()

    train_loss_avg.add(cost)

    if scheduler and (iteration + 1) % config.get("accum_grad", 1) == 0:
        real_step = (iteration + 1) // config.get("accum_grad", 1)
        num_steps = real_step // config["valInterval"]
        inner_steps = (real_step % config["valInterval"]) / config["valInterval"]

        sche_args = {
            "warmup_epochs": config["warmup_epochs"],
            "min_lr": config["min_lr"],
            "lr": config["optimizer"]["lr"],
            "epochs": (config["num_iter"] // config.get("accum_grad", 1))
            // config["valInterval"],
        }
        adjust_learning_rate(configimizer, inner_steps + num_steps, sche_args)

    return cur_lr, train_loss_avg


def validation(
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
):
    elapsed_time = time.time() - start_time
    # for log
    with open(f"{log_dir}/log_train.txt", "a") as log:
        model.eval()
        with torch.no_grad():
            (
                all_costs,
                img_names,
                valid_loss,
                current_accuracy,
                current_bleu,
                current_norm_ED,
                current_word_ED,
                preds,
                labels,
                _,
                _,
            ) = validation_step(
                model, augment, criterion, valid_loader, converter, config, args, device
            )

        model.train(mode=True)

        update_summary(
            iteration,
            OrderedDict([("loss", train_loss_avg.val().item())]),
            OrderedDict(
                [
                    ("loss", valid_loss.item()),
                    ("acc", current_accuracy),
                    ("wed", current_word_ED),
                    ("bleu", current_bleu if current_bleu else "NaN"),
                ]
            ),
            filename=os.path.join(log_dir, "metric_history.csv"),
            write_header=(iteration + 1) == config["valInterval"],
        )

        # training loss and validation loss
        loss_log = f'[{iteration+1}/{config["num_iter"]}] Train loss: {train_loss_avg.val():0.5f}, Valid loss: {valid_loss:0.5f}, Learning rate: {cur_lr:0.5f}, Elapsed_time: {elapsed_time:0.5f}'
        train_loss_avg.reset()

        current_model_log = f'{"Current_accuracy":17s}: {current_accuracy:0.3f}, {"Current_word_ED":17s}: {current_word_ED:0.2f}'
        if current_bleu:
            current_model_log += f', {"Current bleu":17s}: {current_bleu:0.3f}'

        if current_accuracy <= best_accuracy:
            config["patience"] -= config["valInterval"]
            if config["patience"] <= 0:
                print("Early stopping")
                log.write(
                    f"Early stopping with accuracy doesn`t improve from {best_accuracy}\n"
                )
                sys.exit()

        # keep best accuracy model (on valid dataset)
        if current_norm_ED > best_norm_ED:
            best_norm_ED = current_norm_ED

        if current_word_ED > best_word_ED:
            best_word_ED = current_word_ED

        if current_bleu and (current_bleu > best_bleu):
            best_bleu = current_bleu
            save_checkpoint(
                model,
                configimizer,
                best_accuracy,
                best_norm_ED,
                best_word_ED,
                best_bleu,
                iteration,
                f"{log_dir}/best_bleu.pth",
            )

        if current_accuracy > best_accuracy:
            patience = config["early_stop"]
            best_accuracy = current_accuracy
            save_checkpoint(
                model,
                configimizer,
                best_accuracy,
                best_norm_ED,
                best_word_ED,
                best_bleu,
                iteration,
                f"{log_dir}/best_accuracy.pth",
            )

        # keep track latest checkpoint
        save_checkpoint(
            model,
            configimizer,
            best_accuracy,
            best_norm_ED,
            best_word_ED,
            best_bleu,
            iteration,
            f"{log_dir}/last_checkpoint.pth",
        )

        best_model_log = f'{"Best_accuracy":17s}: {best_accuracy:0.3f}, {"Best_word_ED":17s}: {best_word_ED:0.2f}, {"Best BLEU":17s}: {best_bleu:0.3f}'

        loss_model_log = f"{loss_log}\n{current_model_log}\n{best_model_log}"
        print(loss_model_log)
        log.write(loss_model_log + "\n")

        # show some predicted results
        dashed_line = "-" * 80
        head = f'{"Loss":25s} | {"Name":25s} | {"Ground Truth":25s} | {"Prediction":25s} | T/F'
        predicted_result_log = f"{dashed_line}\n{head}\n{dashed_line}\n"

        sorted_res = sorted(
            list(zip(all_costs, labels, preds, img_names)),
            key=lambda x: x[0],
            reverse=True,
        )

        # Retrieve top 10 worst prediction based on validation loss
        for cost, gt, pred, img_name in sorted_res[:10]:
            predicted_result_log += f"{cost:0.4f} | {img_name:25s} | {gt:25s} | {pred:25s} | {str(pred == gt)}\n"

        predicted_result_log += f"{dashed_line}"
        print(predicted_result_log)
        log.write(predicted_result_log + "\n")

    return best_accuracy, best_bleu, best_norm_ED, best_word_ED

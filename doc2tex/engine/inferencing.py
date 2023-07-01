import time
import torch
import torch.utils.data
import torch.nn.functional as F

from torchtext.data import metrics
from utils.model_utils import Averager
from utils.data_utils import Postprocessing
from modules.metrics import ed
from functools import reduce


def validation_step(
    model,
    augment,
    criterion,
    evaluation_loader,
    converter,
    tokenizer,
    config,
    args,
    device,
):
    """validation or evaluation"""
    n_correct = 0
    norm_ED = 0
    word_ED = 0
    length_of_data = 0
    infer_time = 0
    valid_loss_avg = Averager()

    all_loss = []
    total_pred_tokens, total_truth_tokens = [], []
    total_names, total_labels, total_preds = [], [], []

    if config["export_csv"]:
        import csv

        eval_data = config["eval_data"].split("/")[-1]
        save_path = (
            f"./result/{config['exp_name']}/{args.log_path[:-4]}_{eval_data}.csv"
        )
        fo = open(save_path, "wt")
        writer = csv.writer(fo)

    for i, (image_tensors, labels, img_names) in enumerate(evaluation_loader):
        if image_tensors is None and labels is None and img_names is None:
            break

        batch_size = image_tensors.size(0)
        length_of_data = length_of_data + batch_size

        assert image_tensors.device.type == device.type

        if augment:
            image_tensors = torch.clamp(image_tensors, min=0.0, max=255.0)
            image_tensors = image_tensors.div(255.0)
            image_tensors = getattr(augment, "normalize")(image_tensors)

        start_time = time.time()
        if "Attn" in config["Prediction"]["name"]:
            text_for_pred = (
                torch.LongTensor(batch_size, config["batch_max_length"] + 1)
                .fill_(0)
                .to(device)
            )
            text_for_loss, length_for_loss = converter.encode(
                labels, batch_max_length=config["batch_max_length"]
            )
            if config["use_amp"]:
                with torch.cuda.amp.autocast():
                    preds_index, preds, _ = model(
                        image_tensors, text_for_pred, is_train=False
                    )
            else:
                preds_index, preds, _ = model(
                    image_tensors, text_for_pred, is_train=False
                )
            forward_time = time.time() - start_time
            infer_time += forward_time
            target = text_for_loss[:, 1:]  # without [GO] Symbol
            costs = criterion(
                preds.contiguous().view(-1, preds.shape[-1]),
                target.contiguous().view(-1),
            )
            costs = costs.view(batch_size, -1).mean(dim=1)
            valid_loss_avg.add(costs)
            all_loss += costs.detach().cpu().numpy().tolist()
            labels = converter.decode(
                text_for_loss[:, 1:], config.get("token_level", "word")
            )
            preds_str = converter.decode(preds_index, config.get("token_level", "word"))

            if tokenizer is not None:
                labels = tokenizer.process_token_invert(labels)
                preds_str = tokenizer.process_token_invert(preds_str)

            truth_tokens = converter.detokenize(text_for_loss[:, 1:])
            pred_tokens = converter.detokenize(preds_index)

            # calculate accuracy & confidence score
            preds_prob = F.softmax(preds, dim=2)
            preds_max_prob, _ = preds_prob.max(dim=2)
            preds_max_prob = preds_max_prob.permute(1, 0)
            preds_max_prob = reduce(
                lambda x, y: x * y,
                preds_max_prob.detach().cpu(),
                torch.zeros((preds_max_prob.shape[1],)),
            ).tolist()
            np_costs = costs.detach().cpu().numpy().tolist()

            for cost, img_name, gt, pred, pred_prob, pred_token, gt_token in zip(
                np_costs,
                img_names,
                labels,
                preds_str,
                preds_max_prob,
                pred_tokens,
                truth_tokens,
            ):
                gt = gt[: gt.find("[s]")]
                pred_EOS = pred.find("[s]")
                pred = pred[:pred_EOS]

                if config.get("postprocess", True):
                    pred = Postprocessing.remove_unused_whitespace(pred)
                    gt = Postprocessing.remove_unused_whitespace(gt)

                if pred == gt:
                    n_correct += 1

                if config["export_csv"]:
                    writer.writerow((cost, img_name, pred, gt, 1 if pred == gt else 0))
                norm_ED += ed.get_single_ED(gt, pred)
                word_ED += ed.get_word_NED(pred, gt)
                total_names.append(img_name)
                total_labels.append(gt)
                total_preds.append(pred)
                total_pred_tokens.append(pred_token)
                total_truth_tokens.append(gt_token)

        elif config["Prediction"]["name"] == "TFM":
            text_for_pred = (
                torch.LongTensor(batch_size, 1).fill_(converter.dict["[GO]"]).to(device)
            )
            text_for_loss, length_for_loss = converter.encode(
                labels, batch_max_length=config["batch_max_length"]
            )
            if config["use_amp"]:
                with torch.cuda.amp.autocast():
                    preds_index, preds, _ = model(image_tensors, text_for_pred)
            else:
                preds_index, preds, _ = model(image_tensors, text_for_pred)

            forward_time = time.time() - start_time
            infer_time += forward_time
            target = text_for_loss[:, 1:]  # without [GO] Symbol
            costs = criterion(
                preds.contiguous().view(-1, preds.shape[-1]),
                target.contiguous().view(-1),
            )
            costs = costs.view(batch_size, -1).mean(dim=1)
            valid_loss_avg.add(costs)
            all_loss += costs.detach().cpu().numpy().tolist()

            labels = converter.decode(
                text_for_loss[:, 1:], config.get("token_level", "word")
            )
            preds_str = converter.decode(preds_index, config.get("token_level", "word"))

            if tokenizer is not None:
                labels = tokenizer.process_token_invert(labels)
                preds_str = tokenizer.process_token_invert(preds_str)

            truth_tokens = converter.detokenize(text_for_loss[:, 1:])
            pred_tokens = converter.detokenize(preds_index)
            # calculate accuracy & confidence score
            preds_prob = F.softmax(preds, dim=2)
            preds_max_prob, _ = preds_prob.max(dim=2)
            preds_max_prob = preds_max_prob.permute(1, 0)
            preds_max_prob = reduce(
                lambda x, y: x * y,
                preds_max_prob.detach().cpu(),
                torch.zeros((preds_max_prob.shape[1],)),
            ).tolist()
            np_costs = costs.detach().cpu().numpy().tolist()

            for cost, img_name, gt, pred, pred_prob, pred_token, gt_token in zip(
                np_costs,
                img_names,
                labels,
                preds_str,
                preds_max_prob,
                pred_tokens,
                truth_tokens,
            ):
                gt = gt[: gt.find("[s]")]
                pred_EOS = pred.find("[s]")
                pred = pred[:pred_EOS]

                if config.get("postprocess", True):
                    pred = Postprocessing.remove_unused_whitespace(pred)
                    gt = Postprocessing.remove_unused_whitespace(gt)

                if pred == gt:
                    n_correct += 1

                if config["export_csv"]:
                    writer.writerow((cost, img_name, pred, gt, 1 if pred == gt else 0))
                norm_ED += ed.get_single_ED(gt, pred)
                word_ED += ed.get_word_NED(pred, gt)
                total_names.append(img_name)
                total_labels.append(gt)
                total_preds.append(pred)
                total_pred_tokens.append(pred_token)
                total_truth_tokens.append(gt_token)

        if config["sanity_check"]:
            break

    accuracy = n_correct / float(length_of_data)
    norm_ED = norm_ED / float(length_of_data)  # ICDAR2019 Normalized Edit Distance
    word_ED = word_ED / float(length_of_data)

    if config.get("token_level", "word") == "word":
        bleu_score = metrics.bleu_score(
            total_pred_tokens, [[s] for s in total_truth_tokens]
        )
    else:
        bleu_score = None

    if config["export_csv"]:
        fo.close()

    return (
        all_loss,
        total_names,
        valid_loss_avg.val(),
        accuracy,
        bleu_score,
        norm_ED,
        word_ED,
        total_preds,
        total_labels,
        infer_time,
        length_of_data,
    )

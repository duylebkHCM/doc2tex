import os
import sys
import time
import random
import argparse
import yaml
import numpy as np
import pandas as pd
from pathlib import Path

import torch
import torch.utils.data
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import Dataset
from torchtext.data import metrics
from nltk.metrics.distance import edit_distance

sys.path.insert(0, str(Path(__file__).resolve().parent))
from doc2tex.modules.converter import builder
from doc2tex.modules.build_model import Model
from doc2tex.modules.metrics import ed
from doc2tex.utils.predict_utils import resize
from doc2tex.utils.data_utils import Postprocessing
from doc2tex.utils.model_utils import load_checkpoint
from doc2tex.data.data_const import LABEL_KEY
import psutil


class TestDatasetSingle(Dataset):
    def __init__(self, df, opt, start_idx=0, has_label=False):
        test_df = df.copy()
        test_df = test_df.iloc[start_idx:]
        test_df.reset_index(drop=True, inplace=True)
        self.df = test_df
        self.opt = opt
        self.has_label = has_label
        self.resizer = None
        self.preprocess_time = 0

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if idx == len(self):
            return None, None, None

        if self.has_label:
            img_name, label = (
                self.df.loc[idx, LABEL_KEY.IMAGE_ID.value],
                self.df.loc[idx, LABEL_KEY.LABEL.value],
            )
            if len(label):
                if self.opt.get("token_level", "word") == "word":
                    label = [str(label).strip().split()]
                else:
                    label = [str(label)]
        else:
            img_name = self.df.loc[idx, LABEL_KEY.IMAGE_ID.value]

        img_path = os.path.join(self.opt["eval_data"], img_name)
        start_time = time.time()
        new_img = resize(self.resizer, img_path, self.opt)
        end_time = time.time()
        pre_time = end_time - start_time
        self.preprocess_time += pre_time

        if self.has_label:
            return new_img, label, [img_name]
        else:
            return new_img, [img_name]


def run_infer(model, evaluation_loader, converter, tokenizer, config, args):
    total_pred_tokens, total_truth_tokens = [], []
    n_correct = 0
    norm_ED = 0
    word_ED = 0
    length_of_data = 0
    infer_time = 0
    postprocress_time = 0
    memorys_used = []

    if config["export_csv"]:
        import csv

        eval_data = config["eval_data"].split("/")[-1]
        save_path = f"./result/{config['problem']}/{config['exp_name']}/{args.log_path[:-4]}_{eval_data}.csv"
        if args.start_idx == 0:
            fo = open(save_path, "wt")
        else:
            fo = open(save_path, "at")
        writer = csv.writer(fo)

    for _, (image_tensors, labels, img_names) in enumerate(evaluation_loader):
        if image_tensors is None and labels is None and img_names is None:
            break

        if config.get("data_filtering", True):
            if isinstance(labels, str):
                continue
            if len(labels) > config["batch_max_length"]:
                continue

        batch_size = image_tensors.size(0)
        length_of_data = length_of_data + batch_size
        image = image_tensors.to(config["device"])

        start_time = time.time()

        if "Attn" in config["Prediction"]["name"]:
            text_for_pred = (
                torch.LongTensor(batch_size, config["batch_max_length"] + 1)
                .fill_(0)
                .to(config["device"])
            )
            if isinstance(labels, list):
                text_for_loss, _ = converter.encode(
                    labels, batch_max_length=config["batch_max_length"]
                )
            if config["device"] == "cuda" and config["use_amp"]:
                with torch.cuda.amp.autocast():
                    preds_index, preds, _ = model(
                        image, text_for_pred, is_train=False, is_test=True
                    )
            else:
                preds_index, preds, _ = model(
                    image, text_for_pred, is_train=False, is_test=True
                )
            forward_time = time.time() - start_time
            infer_time += forward_time
            if isinstance(labels, list):
                gt = converter.decode(
                    text_for_loss[:, 1:], config.get("token_level", "word")
                )[0]
            else:
                gt = ""
            pred = converter.decode(preds_index, config.get("token_level", "word"))[0]

            if tokenizer is not None:
                if isinstance(labels, list):
                    labels = tokenizer.process_token_invert(labels)
                pred = tokenizer.process_token_invert(pred)

            if isinstance(labels, list):
                truth_token = converter.detokenize(text_for_loss[:, 1:])[0]
            pred_token = converter.detokenize(preds_index)[0]

        elif "TFM" in config["Prediction"]["name"]:
            text_for_pred = (
                torch.LongTensor(batch_size, 1)
                .fill_(converter.dict["[GO]"])
                .to(config["device"])
            )
            text_for_loss, length_for_loss = converter.encode(
                labels, batch_max_length=config["batch_max_length"]
            )
            if config["device"] == "cuda" and config["use_amp"]:
                with torch.cuda.amp.autocast():
                    preds_index, preds = model(image, text_for_pred, is_test=True)
            else:
                preds_index, preds = model(image, text_for_pred, is_test=True)
            forward_time = time.time() - start_time
            infer_time += forward_time
            gt = converter.decode(
                text_for_loss[:, 1:], config.get("token_level", "word")
            )[0]
            pred = converter.decode(preds_index, config.get("token_level", "word"))[0]

            if tokenizer is not None:
                labels = tokenizer.process_token_invert(labels)
                pred = tokenizer.process_token_invert(pred)

            truth_token = converter.detokenize(text_for_loss[:, 1:])[0]
            pred_token = converter.detokenize(preds_index)[0]

        if config["beam_size"] == 1:
            preds_prob = F.log_softmax(preds, dim=2)[0]
            pred_max_prob = preds_prob.max(dim=1)[0]

        if (
            "Attn" in config["Prediction"]["name"]
            or "TFM" in config["Prediction"]["name"]
        ):
            if isinstance(labels, list):
                gt = gt[: gt.find("[s]")]
            pred_EOS = pred.find("[s]")
            pred = pred[:pred_EOS]
            if config["beam_size"] == 1:
                pred_max_prob = pred_max_prob[:pred_EOS]

        if config.get("postprocess", True):
            start_pos = time.time()
            pred = Postprocessing.remove_unused_whitespace(pred)
            post_time = time.time() - start_pos
            postprocress_time += post_time
            gt = Postprocessing.remove_unused_whitespace(gt)

        if isinstance(labels, list):
            if pred == gt:
                n_correct += 1
                iscorrect = 1
            else:
                iscorrect = 0
        else:
            n_correct += 1
            iscorrect = 1

        if isinstance(labels, list):
            total_truth_tokens.append(truth_token)
        else:
            total_truth_tokens.append([""])
        total_pred_tokens.append(pred_token)

        # ICDAR2019 Normalized Edit Distance
        if len(gt) == 0 or len(pred) == 0:
            cur_ED = 0
        elif len(gt) > len(pred):
            cur_ED = 1 - edit_distance(pred, gt) / len(gt)
        else:
            cur_ED = 1 - edit_distance(pred, gt) / len(pred)

        norm_ED += cur_ED
        cur_word_ED = ed.get_word_NED(pred, gt)
        word_ED += cur_word_ED

        cur_bleu = metrics.bleu_score(
            candidate_corpus=[pred_token], references_corpus=[[truth_token]]
        )

        if config["export_csv"]:
            if args.strong_log:
                writer.writerow(
                    (img_names[0], pred, gt, cur_ED, cur_word_ED, cur_bleu, iscorrect)
                )
            else:
                writer.writerow((img_names[0], pred, gt, iscorrect))

        if torch.cuda.is_available():
            memory_used = int(torch.cuda.max_memory_allocated() / (1024.0 * 1024.0))
        else:
            memory_used = int(psutil.virtual_memory()[3] / (1024.0 * 1024.0))
        memorys_used.append(memory_used)

    accuracy = n_correct / float(length_of_data) * 100
    norm_ED = norm_ED / float(length_of_data)  # ICDAR2019 Normalized Edit Distance
    word_ED = word_ED / float(length_of_data)
    if config.get("token_level", "word") == "word":
        bleu_score = metrics.bleu_score(
            total_pred_tokens, [[s] for s in total_truth_tokens]
        )
    else:
        bleu_score = None

    avg_mem_used = sum(memorys_used) / float(length_of_data)
    if config["export_csv"]:
        fo.close()

    return (
        accuracy,
        bleu_score,
        norm_ED,
        word_ED,
        avg_mem_used,
        infer_time,
        postprocress_time,
        length_of_data,
    )


def infer(config, args, tokenizer):
    converter = builder.create_converter(config, config["device"])
    config["num_class"] = len(converter.character)

    model = Model(config)
    load_checkpoint(config, model)

    params_num = []
    for p in filter(lambda p: p.requires_grad, model.parameters()):
        params_num.append(np.prod(p.size()))

    # resizer = prepare_resizer(config) if config['use_resizer'] else None
    resizer = None

    if args.console:
        with torch.no_grad():
            df = pd.read_csv(
                args.csv_dir, sep=LABEL_KEY.DELIMITER.value, keep_default_na=False
            )
            dataset = TestDatasetSingle(
                df, config, start_idx=args.start_idx, has_label=True
            )
            dataset.resizer = resizer
            (
                accuracy_by_best_model,
                bleu,
                norm_ED,
                word_ED,
                avg_mem_used,
                infer_time,
                postpro_time,
                length_of_data,
            ) = run_infer(model, dataset, converter, tokenizer, config, args)
    else:
        config["exp_name"] = "_".join(config["saved_model"].split("/")[2:])

        """ keep evaluation model and result logs """
        os.makedirs(f'./result/{config["exp_name"]}', exist_ok=True)

        """ evaluation """
        model.eval()
        with torch.no_grad():
            log = open(f'./result/{config["exp_name"]}/{args.log_path}', "w")
            df = pd.read_csv(
                args.csv_dir, sep=LABEL_KEY.DELIMITER.value, keep_default_na=False
            )
            dataset = TestDatasetSingle(
                df, config, start_idx=args.start_idx, has_label=True
            )
            dataset.resizer = resizer

            (
                accuracy_by_best_model,
                bleu,
                norm_ED,
                word_ED,
                avg_mem_used,
                infer_time,
                postpro_time,
                length_of_data,
            ) = run_infer(model, dataset, converter, tokenizer, config, args)

            print(f"Acc: {accuracy_by_best_model:0.3f}")
            if bleu:
                print(f"BLEU-4: {bleu:0.5f}")
            print(f"Norm Edit Distance: {norm_ED:0.5f}")
            print(f"Symbol Match (Word Edit Distance): {word_ED:0.5f}")
            print(f"Infer time {infer_time} s")
            print(f"Avg infer time {infer_time / float(length_of_data)} s")
            print(f"Preprocess time: {dataset.preprocess_time} s")
            print(f"Avg pre time: {dataset.preprocess_time / float(length_of_data)}")
            print(f"Postprocess time: {postpro_time} s")
            print(f"Avg post time {postpro_time / float(length_of_data)} s")
            print(f"Memory used: {avg_mem_used} MB\n")
            log.write(f"Trainable params num: {str(sum(params_num))}\n")
            log.write(f"Acc: {accuracy_by_best_model:0.3f}\n")
            if bleu:
                log.write(f"BLEU-4: {bleu:0.5f}\n")
            log.write(f"Norm Edit Distance: {norm_ED:0.5f}\n")
            log.write(f"Symbol Match (Word Edit Distance): {word_ED:0.5f}\n")
            log.write(f"Total Infer Time: {infer_time} s\n")
            log.write(f"Avg Infer Time: {infer_time / float(length_of_data)} s\n")
            log.write(f"Postprocess time: {postpro_time} s\n")
            log.write(f"Avg post time {postpro_time / float(length_of_data)} s\n")
            log.write(f"Memory used: {avg_mem_used} MB\n")
            log.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to config yaml file")
    parser.add_argument("--csv_dir", required=True, help="Path to csv file")
    parser.add_argument(
        "--start_idx", type=int, default=0, help="Index to start from csv file"
    )
    parser.add_argument(
        "--data_dir", required=True, help="Path to image folder to infer"
    )
    parser.add_argument("--amp", type=bool, default=False)
    parser.add_argument("--resizer", action="store_true", default=False)
    parser.add_argument(
        "--log_path", required=True, help="Path to save evaluation result"
    )
    parser.add_argument(
        "--batch_size",
        required=True,
        type=int,
        help="test on batch or with single sample",
    )
    parser.add_argument("--num_workers", type=int, default=-1, help="number of workers")
    parser.add_argument(
        "--strong_log",
        action="store_true",
        default=False,
        help="additionally log more metric in csv file",
    )
    parser.add_argument("--console", default=False)
    args = parser.parse_args()

    config = yaml.load(open(args.config), Loader=yaml.FullLoader)

    config["batch_size"] = args.batch_size
    config["workers"] = args.num_workers
    config["use_amp"] = args.amp and torch.cuda.is_available()
    config["use_resizer"] = args.resizer
    config["eval_data"] = args.data_dir

    random.seed(config["manualSeed"])
    np.random.seed(config["manualSeed"])
    torch.manual_seed(config["manualSeed"])

    if torch.cuda.is_available():
        torch.cuda.manual_seed(config["manualSeed"])
        cudnn.benchmark = False
        cudnn.deterministic = False
        config["num_gpu"] = torch.cuda.device_count()
        config["device"] = "cuda"
    else:
        config["num_gpu"] = 0
        config["device"] = "cpu"

    if config["num_gpu"] > 1:
        config["workers"] = config["num_gpu"] * config["workers"]
        config["batch_size"] = config["num_gpu"] * config["batch_size"]

    infer(config, args)

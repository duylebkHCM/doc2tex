import os
from pathlib import Path
from matplotlib import pyplot as plt
import numpy as np
import torch
from torch.nn import functional as F
import random
import yaml
import sys
from ast import literal_eval
import re
import pandas as pd
import cv2
import copy

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from doc2tex.modules.build_model import Model
from doc2tex.utils.predict_utils import resize
from doc2tex.utils.model_utils import load_checkpoint
from modules.converter import builder
from .config import CFG
from PIL import ImageFont, ImageDraw, Image


class VizTool:
    @staticmethod
    def get_saliency_map(image, saliency_map):
        """
        Save saliency map on image.

        Args:
            image: Tensor of size (3,H,W)
            saliency_map: Tensor of size (1,H,W)
            filename: string with complete path and file extension

        """
        saliency_map = saliency_map - saliency_map.min()
        saliency_map = saliency_map / saliency_map.max()
        saliency_map = saliency_map.clip(0, 1)
        saliency_map = np.expand_dims(saliency_map, axis=-1)

        saliency_map = np.uint8(saliency_map * 255)

        # Apply JET colormap
        color_heatmap = cv2.applyColorMap(saliency_map, cv2.COLORMAP_JET)

        # Combine image with heatmap
        img_with_heatmap = np.float32(color_heatmap) + np.float32(image)
        img_with_heatmap = img_with_heatmap / np.max(img_with_heatmap)

        return np.uint8(255 * img_with_heatmap)

    @staticmethod
    def show_mask_on_image(img, mask):
        mask = np.clip(mask, a_min=0.0, a_max=1.0)
        print("Max mask", np.max(mask))
        print("Min mask", np.min(mask))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.float32(img) / 255
        heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
        heatmap = np.float32(heatmap) / 255
        cam = heatmap + np.float32(img)
        cam = (cam - np.min(cam)) / (np.max(cam) - np.min(cam))
        return np.uint8(255 * cam)


def visualize_att(
    image_path,
    output_dir,
    seq,
    pred_seq,
    gt_label,
    alphas,
    feat_pad,
    rev_word_map,
    smooth=True,
    down_sample=2,
    num_samples=12,
    put_text=False,
    display_last=False,
    **kwargs,
):
    def get_mask(image, current_alpha):
        current_alpha = alpha = F.interpolate(
            current_alpha.unsqueeze(0).unsqueeze(0),
            scale_factor=2,
            mode="bilinear",
            align_corners=True,
        )
        current_alpha = current_alpha.squeeze(0).squeeze(0)
        alpha_h, alpha_w = (
            current_alpha.shape[0] - feat_pad[1],
            current_alpha.shape[1] - feat_pad[0],
        )
        current_alpha = current_alpha[:alpha_h, :alpha_w]
        final_h = int(16 * (current_alpha.shape[0] + 1) / down_sample)
        final_w = int(4 * (current_alpha.shape[1] - 1) / down_sample)

        image = image.resize((final_w, final_h), Image.BICUBIC)

        alpha = F.interpolate(
            current_alpha.unsqueeze(0).unsqueeze(0),
            (final_h, final_w),
            mode="bilinear",
            align_corners=True,
        )
        alpha = alpha.squeeze(0).squeeze(0)
        alpha = np.asarray(alpha).astype(float)
        np_image = np.asarray(image).astype("uint8")
        mask = VizTool.get_saliency_map(np_image, alpha)
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
        return mask

    save_fig_seperate = kwargs.pop("save_seperate_fig", False)

    image = Image.open(image_path).convert("RGB")

    words = [""]
    words += [rev_word_map[ind] for ind in seq]
    alphas = torch.cat(
        [torch.zeros_like(alphas[:1, ...], device=alphas.device), alphas], dim=0
    )
    assert len(alphas) == len(words), f"{len(alphas)} vs {len(words)}"

    if num_samples != -1:
        assert num_samples % 2 == 0
        assert num_samples <= len(alphas)

        if display_last:
            word_lst = words[-num_samples:]
            alpha_lst = alphas[-num_samples:]
        else:
            word_lst = words[:num_samples]
            alpha_lst = alphas[:num_samples]
    else:
        word_lst = words
        alpha_lst = alphas

    if save_fig_seperate:
        cur_idx = 0
        img_name = os.path.basename(image_path).split(".")[0]
        save_path = os.path.join(output_dir, img_name)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        for word, current_alpha in zip(word_lst, alpha_lst):
            if cur_idx == 0:
                cv2_image = np.asarray(image)
                cv2_image = cv2.cvtColor(cv2_image, cv2.COLOR_RGB2BGR)
                cv2.imwrite(os.path.join(save_path, img_name + ".png"), cv2_image)
            else:
                resize_mask = get_mask(image, current_alpha)

                if put_text:
                    padd_mask = np.ones(
                        (
                            resize_mask.shape[0] + 40,
                            resize_mask.shape[1],
                            resize_mask.shape[2],
                        )
                    ).astype("uint8")
                    padd_mask = padd_mask * 255
                    padd_mask[40:, ...] = resize_mask
                    pil_mask = Image.fromarray(padd_mask)
                    draw = ImageDraw.Draw(pil_mask)
                    font = ImageFont.truetype("fonts/arial.ttf", size=25)
                    draw.text(
                        (int(pil_mask.size[0] // 2), 10),
                        word,
                        fill=(0, 0, 0),
                        font=font,
                    )
                    padd_mask = cv2.cvtColor(np.asarray(pil_mask), cv2.COLOR_RGB2BGR)
                else:
                    padd_mask = cv2.cvtColor(resize_mask, cv2.COLOR_RGB2BGR)
                cv2.imwrite(
                    os.path.join(
                        save_path, img_name + "_" + str(cur_idx).zfill(2) + ".png"
                    ),
                    padd_mask,
                )
            cur_idx += 1
    else:
        n_rows = np.ceil(len(word_lst) / 2.0).astype("int")
        fig, ax = plt.subplots(n_rows, 2, constrained_layout=True)

        title_ = (
            f"Attention weight for each step in decoding process\n"
            f"Pred latex: {pred_seq}\n"
            f"Label latex: {gt_label}\n"
            f"Correct: {pred_seq == gt_label}"
        )

        fig.suptitle(title_)

        cur_idx = 0
        for word, current_alpha in zip(word_lst, alpha_lst):
            if cur_idx == 0:
                ax[cur_idx // 2, cur_idx % 2].imshow(image, cmap="gray")
            else:
                resize_mask = get_mask(image, current_alpha)
                ax[cur_idx // 2, cur_idx % 2].imshow(resize_mask)

            if cur_idx > 0:
                ax[cur_idx // 2, cur_idx % 2].set_title(
                    word, fontdict={"fontsize": 13, "fontweight": "medium"}
                )

            ax[cur_idx // 2, cur_idx % 2].axis("off")

            cur_idx += 1

        plt.show()
        img_name = os.path.basename(image_path).split(".")[0]
        output_path = os.path.join(output_dir, "weight_attn_" + img_name + ".png")
        fig.savefig(output_path)


def get_test_sample(img_dir, csv_dir, condition=None):
    df = pd.read_csv(csv_dir)
    len_pattn = re.search(r"(\(.*\)).*", condition)
    df["len"] = df["pred"].apply(lambda x: len(x.strip().split()))

    if os.path.isdir(img_dir):
        if len_pattn is None:
            len_con = "len == " + str(
                random.randint(df["len"].min(), df["len"].max() + 1)
            )
        else:
            len_con = len_pattn.group(1)[1:-1]

        iscorrect_pattn = re.search(r".*%iscorrect: (\w+)", condition)
        if iscorrect_pattn is None:
            iscorrect = True
        else:
            iscorrect = iscorrect_pattn.group(1)

        sub_df = df.query(len_con)
        sub_df = sub_df[sub_df["iscorrect"] == int(literal_eval(iscorrect))]["name"]
        img_lst = sub_df.values.tolist()
        if len(img_lst) == 0:
            return None, None, None
        else:
            random_img = random.choice(img_lst)
            img_path = os.path.join(img_dir, random_img)

            gt_label = df[df["name"] == random_img]["label"].values.tolist()
            pred_str = df[df["name"] == random_img]["pred"].values.tolist()

            gt_label = gt_label[0]
            pred_str = pred_str[0]
            print("PRED LEN", len(pred_str.strip().split()))
            print("GT LABEL LEN", len(gt_label.strip().split()))
            print("GT LABEL", gt_label)

            return gt_label, pred_str, img_path
    else:
        df = df[df["name"] == os.path.basename(img_dir)]
        gt_label = df["label"].values.tolist()
        pred_str = df["pred"].values.tolist()

        gt_label = gt_label[0]
        pred_str = pred_str[0]
        print("PRED LEN", len(pred_str.strip().split()))
        print("GT LABEL LEN", len(gt_label.strip().split()))
        print("GT LABEL", gt_label)

        return gt_label, pred_str, img_dir


def prepare_pred(config, img_path):
    config["use_resizer"] = False
    converter = builder.create_converter(config, config["device"])
    config["num_class"] = len(converter.character)
    model = Model(config)
    load_checkpoint(config, model)
    ts_img = resize(None, img_path, config)
    return model, ts_img, converter


def get_pred(model, ts_img, converter):
    text_for_pred = torch.LongTensor(1, config["batch_max_length"] + 1).fill_(0)

    model.eval()
    print("Tensor img shape", ts_img.shape)
    with torch.no_grad():
        preds_index, _, add_on = model(ts_img, text_for_pred, is_train=False)

    pred_index: list = preds_index[0].tolist()
    decoder_attn = add_on.get("decoder_attn", None)
    pad_info = add_on.get("feat_pad", None)

    assert len(decoder_attn) == len(pred_index)

    pred_EOS = pred_index.index(1)
    pred_index = pred_index[:pred_EOS]
    decoder_attn = decoder_attn[:pred_EOS, ...]
    print("Max attn", decoder_attn.max())
    print("Min attn", decoder_attn.min())

    rev_map = {v: k for k, v in converter.dict.items()}
    pred_str = converter.decode(preds_index, token_level="word")[0]
    pred_EOS = pred_str.find("[s]")
    pred_str = pred_str[:pred_EOS]

    return rev_map, pred_index, pred_str, decoder_attn, pad_info


def visualize_decoder(
    img_dir,
    csv_dir,
    output_dir,
    config,
    condition,
    down_sample,
    num_samples,
    smooth,
    put_text,
    display_last,
    **kwargs,
):
    gt_label, pred_str, img_path = get_test_sample(img_dir, csv_dir, condition)

    if gt_label is None or pred_str is None:
        raise ValueError("condition do not exist")

    model, ts_img, converter = prepare_pred(config, img_path)
    rev_map, pred_index, pred_str, decoder_attn, feat_pad = get_pred(
        model, ts_img, converter
    )

    if decoder_attn is not None:
        print("decoder_attn", decoder_attn.shape)
        visualize_att(
            img_path,
            output_dir,
            pred_index,
            pred_str,
            gt_label,
            decoder_attn,
            feat_pad,
            rev_map,
            smooth=smooth,
            down_sample=down_sample,
            num_samples=num_samples,
            put_text=put_text,
            display_last=display_last,
            **kwargs,
        )
    else:
        print("Do not have attn weight to viz")


if __name__ == "__main__":
    config = yaml.load(open("predict.yaml"), Loader=yaml.FullLoader)
    config["device"] = "cpu"
    print(CFG.get_attr(CFG.Decoder.__dict__))
    attrs = CFG.get_attr(CFG.Decoder.__dict__)
    if isinstance(CFG.image_name, list):
        img_dirs = [os.path.join(CFG.img_dir, img_name) for img_name in CFG.image_name]
        for img_dir in img_dirs:
            Decoder_cur_cfg = copy.deepcopy(config)
            visualize_decoder(
                img_dir, CFG.csv_dir, CFG.output_dir, Decoder_cur_cfg, **attrs
            )
    elif isinstance(CFG.image_name, str):
        img_dir = os.path.join(CFG.img_dir, CFG.image_name)
        visualize_decoder(img_dir, CFG.csv_dir, CFG.output_dir, config, **attrs)
    else:
        visualize_decoder(CFG.img_dir, CFG.csv_dir, CFG.output_dir, config, **attrs)

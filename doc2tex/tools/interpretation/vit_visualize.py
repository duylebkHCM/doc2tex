import numpy as np
import cv2
import torch
from PIL import Image
from pathlib import Path

from torch.nn import functional as F
import pandas as pd
import os
import random
import re
from ast import literal_eval
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from doc2tex.modules.build_model import Model
from doc2tex.utils.predict_utils import resize
from doc2tex.utils.model_utils import load_checkpoint
from modules.converter import builder
from .config import CFG

from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam import GradCAM


class VITAttentionRollout:
    def __init__(
        self,
        model,
        debug,
        attention_layer_name="attn_drop",
        head_fusion="max",
        discard_ratio=0.9,
    ):
        self.model = model
        self.discard_ratio = discard_ratio
        for name, module in self.model.named_modules():
            if attention_layer_name in name:
                module.register_forward_hook(self.get_attention)

        self.attentions = []
        self.head_fusion = head_fusion
        self.debug = debug

    def get_attention(self, module, input, output):
        if self.debug:
            print("Output", output.shape)
        # output shape (LxL)
        self.attentions.append(output.cpu())

    def rollout(self, embed_width, embed_height):
        result = torch.eye(self.attentions[0].size(-1))  # shape LxL

        with torch.no_grad():
            for attention in self.attentions:
                if self.head_fusion == "mean":
                    attention_heads_fused = attention.mean(axis=1)
                elif self.head_fusion == "max":
                    attention_heads_fused = attention.max(axis=1)[0]
                elif self.head_fusion == "min":
                    attention_heads_fused = attention.min(axis=1)[0]
                else:
                    raise "Attention head fusion type Not supported"

                # Drop the lowest attentions, but
                # don't drop the class token
                flat = attention_heads_fused.view(attention_heads_fused.size(0), -1)
                print("flat sha[e", flat.shape)
                _, indices = flat.topk(
                    int(flat.size(-1) * self.discard_ratio), -1, False
                )
                print("indices", indices)
                # indices = indices[indices != 0]
                flat[0, indices] = 0

                I = torch.eye(attention_heads_fused.size(-1))
                a = (attention_heads_fused + 1.0 * I) / 2
                a = a / a.sum(dim=-1)
                result = torch.matmul(a, result)

        # Look at the total attention between the class token,
        # and the image patches
        mask = result[0, 0, 1:]

        mask = mask.reshape(embed_height, embed_width).numpy()
        mask = mask / np.max(mask)
        return mask

    def __call__(self, input_tensor):
        self.attentions = []

        with torch.no_grad():
            context_feature, output_shape = self.model.forward_encoder(
                input_tensor
            )  # [B, L, C]
            # output_index = translate()

        if self.debug:
            print("context_feature", context_feature.shape)
            print("output_shape", output_shape)

        return self.rollout(output_shape[1], output_shape[0])


def get_test_sample(img_dir, csv_dir, condition=None):
    df = pd.read_csv(csv_dir)
    len_pattn = re.search(r"(\(.*\)).*", condition)
    df["len"] = df["pred"].apply(lambda x: len(x.strip().split()))

    if "name" not in condition:
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
        df = df[df["name"] == condition.replace("name", "")]["name"]
        img_lst = df.values.tolist()
        gt_label = df[df["name"] == random_img]["label"].values.tolist()
        pred_str = df[df["name"] == random_img]["pred"].values.tolist()

        gt_label = gt_label[0]
        pred_str = pred_str[0]
        print("PRED LEN", len(pred_str.strip().split()))
        print("GT LABEL LEN", len(gt_label.strip().split()))
        print("GT LABEL", gt_label)

        return gt_label, pred_str, img_path


def prepare_pred(config, img_path):
    config["use_resizer"] = False
    converter = builder.create_converter(config)
    config["num_class"] = len(converter.character)
    model = Model(config)
    load_checkpoint(config, model)
    ts_img = resize(None, img_path, config)
    return model, ts_img, converter


def get_fe_pred(model, ts_img, config):
    target_layers = [
        getattr(model.SequenceModeling.patch_embed, config["target_layers"])
    ]
    with GradCAM(model=model, target_layers=target_layers, use_cuda=False) as cam:
        grayscale_cam = cam(
            input_tensor=ts_img, targets=None, aug_smooth=False, eigen_smooth=False
        )
        grayscale_cam = grayscale_cam[0, :]
    return grayscale_cam


def get_pred(model, ts_img, config):
    assert ts_img.shape[0] == 1

    grad_rollout = VITAttentionRollout(
        model,
        discard_ratio=config["discard_ratio"],
        debug=config["debug"],
        head_fusion=config["head_fusion"],
    )

    mask = grad_rollout(ts_img)

    return mask


def get_seqmodel_output(model, ts_img, get_mean, config):
    with torch.no_grad():
        context_feature, output_shape = model.forward_encoder(ts_img)  # [B, L, C]
    print(context_feature.shape)
    print(output_shape)
    context_feature = context_feature[:, 1:, :].reshape(
        1, output_shape[0], output_shape[1], 256
    )
    if get_mean:
        context_feature = (
            context_feature.mean(dim=-1, keepdims=True).squeeze(dim=-1).squeeze(dim=0)
        )
        print("seqmodel output", context_feature.min(), context_feature.max())
    else:
        context_feature = context_feature.squeeze(dim=0)
    return context_feature


def get_fe_output(model, ts_img, config):
    with torch.no_grad():
        context_feature, output_shape, _ = model.SequenceModeling.patch_embed(
            ts_img
        )  # [B, L, C]
    print(context_feature.shape)
    # print(output_shape)
    context_feature = context_feature.reshape(
        1, output_shape["height"] // 2, output_shape["width"] // 2, 256
    )
    context_feature = (
        context_feature.mean(dim=-1, keepdims=True).squeeze(dim=-1).squeeze(dim=0)
    )
    print("fe output", context_feature.min(), context_feature.max())
    return context_feature


def show_mask_on_image(img, mask: torch.FloatTensor):
    img = np.float32(img) / 255
    print("mask min", mask.min().item())
    print("mask max", mask.max().item())
    mask = np.clip(mask.numpy(), a_min=0.0, a_max=1.0)
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + np.float32(img)
    cam = cam / np.max(cam)
    return np.uint8(255 * cam)


def get_saliency_map(image, saliency_map):
    """
    Save saliency map on image.

    Args:
        image: Tensor of size (3,H,W)
        saliency_map: Tensor of size (1,H,W)
        filename: string with complete path and file extension

    """
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
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


def visualize_vit(img_path, output_dir, mask, name, down_sample=2):
    image = Image.open(img_path).convert("RGB")
    w, h = image.size
    new_h = int(h / down_sample)
    new_w = int(w / down_sample)
    if down_sample > 1:
        image = image.resize((new_w, new_h), Image.LANCZOS)
    else:
        image = image.resize((new_w, new_h), Image.BICUBIC)

    np_img = np.asarray(image).astype("uint8")
    # mask = cv2.resize(mask, (np_img.shape[1], np_img.shape[0]))
    mask = torch.from_numpy(mask).float()
    resize_mask = F.interpolate(
        mask.unsqueeze(0).unsqueeze(0),
        (np_img.shape[0], np_img.shape[1]),
        mode="bicubic",
        align_corners=True,
    )
    resize_mask = resize_mask.squeeze(0).squeeze(0)
    resize_mask = np.asarray(resize_mask).astype(float)
    # assert mask.shape[1] == len(pred_seq)
    resize_mask = get_saliency_map(np_img, resize_mask)
    # resize_mask = cv2.cvtColor(resize_mask, cv2.COLOR_RGB2BGR)

    img_name = os.path.basename(img_path).split(".")[0]
    save_path = os.path.join(output_dir, img_name)

    if not os.path.exists(save_path):
        os.makedirs(save_path)
    output_path = save_path + "/" + name + ".png"
    cv2.imwrite(output_path, resize_mask)
    image.save(os.path.join(save_path, os.path.basename(img_path)), format="PNG")


def postprocess_fe(feature_map, img_path, name, down_sample):
    image = Image.open(img_path).convert("RGB")
    w, h = image.size
    new_h = int(h / down_sample)
    new_w = int(w / down_sample)
    if down_sample > 1:
        image = image.resize((new_w, new_h), Image.LANCZOS)
    else:
        image = image.resize((new_w, new_h), Image.BICUBIC)

    np_img = np.asarray(image).astype("uint8")

    if len(feature_map.shape) == 3:
        feature_map = feature_map.view(256, feature_map.shape[0], feature_map.shape[1])
        for idx, f_map in enumerate(feature_map):
            resize_mask = F.interpolate(
                f_map.unsqueeze(0).unsqueeze(0),
                (np_img.shape[0], np_img.shape[1]),
                mode="bicubic",
                align_corners=True,
            )
            resize_mask = resize_mask.squeeze(0).squeeze(0)
            resize_mask = np.asarray(resize_mask).astype(float)

            resize_mask = resize_mask - resize_mask.min()
            resize_mask = resize_mask / resize_mask.max()
            resize_mask = np.expand_dims(resize_mask, axis=-1)

            resize_mask = np.uint8(resize_mask * 255)
            img_name = os.path.basename(img_path).split(".")[0]
            save_path = os.path.join(output_dir, img_name)
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            if not os.path.exists(os.path.join(save_path, "seq_split")):
                os.makedirs(os.path.join(save_path, "seq_split"))
            output_path = (
                save_path + "/" + "seq_split" + "/" + name + "_" + str(idx) + ".png"
            )
            cv2.imwrite(output_path, resize_mask)
    else:
        resize_mask = F.interpolate(
            feature_map.unsqueeze(0).unsqueeze(0),
            (np_img.shape[0], np_img.shape[1]),
            mode="bicubic",
            align_corners=True,
        )
        resize_mask = resize_mask.squeeze(0).squeeze(0)
        resize_mask = np.asarray(resize_mask).astype(float)

        resize_mask = resize_mask - resize_mask.min()
        resize_mask = resize_mask / resize_mask.max()
        resize_mask = np.expand_dims(resize_mask, axis=-1)

        resize_mask = np.uint8(resize_mask * 255)
        img_name = os.path.basename(img_path).split(".")[0]
        save_path = os.path.join(output_dir, img_name)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        output_path = save_path + "/" + name + ".png"
        cv2.imwrite(output_path, resize_mask)


def visualize_encoder(img_dir, output_dir, csv_dir, get_mean, config, opt):
    if os.path.isdir(img_dir):
        gt_label, pred_str, img_path = get_test_sample(
            img_dir, csv_dir, config["condition"]
        )
    else:
        img_path = img_dir
    model, ts_img, _ = prepare_pred(config, img_path)
    if opt == "1":
        mask = get_pred(model, ts_img, config)
        visualize_vit(
            img_path,
            output_dir,
            mask,
            "vit_attn_weight",
            down_sample=config["downsample"],
        )

        seq_map = get_seqmodel_output(model, ts_img, get_mean, config)
        postprocess_fe(seq_map, img_path, "seq_map", config["downsample"])

        fe_map = get_fe_output(model, ts_img, config)
        postprocess_fe(fe_map, img_path, "fe_map", config["downsample"])

    else:
        mask = get_fe_pred(model, ts_img, config)
        image = Image.open(img_dir).convert("RGB")
        rgb_img = np.asarray(image, dtype=np.float32) / 255
        cam_image = show_cam_on_image(rgb_img, mask, use_rgb=True)
        cam_image = cv2.cvtColor(cam_image, cv2.COLOR_RGB2BGR)
        img_name = os.path.basename(img_dir).split(".")[0]
        cv2.imwrite(os.path.join(output_dir, img_name), cam_image)


if __name__ == "__main__":
    import yaml

    opt = sys.argv[1]
    config = yaml.load(open("predict.yaml"), Loader=yaml.FullLoader)
    config["device"] = "cpu"
    print(CFG.get_attr(CFG.Encoder.__dict__))
    attrs = CFG.get_attr(CFG.Encoder.__dict__)
    fe_attr = CFG.get_attr(CFG.FE.__dict__)
    if opt == "1":
        config.update(attrs)
        output_dir = CFG.Encoder.output_dir
    else:
        config.update(fe_attr)
        output_dir = CFG.FE.output_dir
    visualize_encoder(CFG.img_dir, output_dir, CFG.csv_dir, True, config, opt)

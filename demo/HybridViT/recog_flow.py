import re
from typing import Any
from collections import OrderedDict
import re

import torch
import torch.nn as nn
from torch.nn import functional as F
from PIL import Image
from timm.models.resnetv2 import ResNetV2

from .recognizers.build_model import Model
from .module.converter import AttnLabelConverter, TFMLabelConverter
from .helper import resize


class MathRecognition(object):
    def __init__(self, config, resizer):
        self.args = config
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.args["device"] = device
        self.device = device
        self._prepare_vocab()
        self.model = self._get_model()
        self.resizer = resizer

    def _mapping_ckpt(self, state_dict):
        new_state_dict = OrderedDict()

        for name, param in state_dict.items():
            if name.startswith("Transformation"):
                continue
            elif name.startswith("FeatureExtraction"):
                new_name = name.replace(
                    "FeatureExtraction", "featextractor.FeatureExtraction"
                )
                new_state_dict[new_name] = param
            elif name.startswith("SequenceModeling"):
                new_name = name.replace(
                    "SequenceModeling", "seqmodeler.SequenceModeling"
                )
                new_state_dict[new_name] = param
            elif name.startswith("Prediction"):
                new_name = name.replace("Prediction", "predicter.Prediction")
                new_state_dict[new_name] = param
            else:
                new_state_dict[name] = param

        return new_state_dict

    def _get_model(self):
        model = Model(self.args)
        state_dict = torch.load(self.args["weight_path"], map_location="cpu")
        new_state_dict = self._mapping_ckpt(state_dict)
        model.load_state_dict(new_state_dict)
        model = model.eval()

        if self.device == "cuda":
            num_gpu = torch.cuda.device_count()
            if num_gpu > 1:
                model = nn.DataParallel(model).to(self.device)
            else:
                model.to(self.device)

        return model

    def _prepare_vocab(self):
        with open(self.args["vocab"], "rt") as f:
            for line in f:
                self.args["character"] += [line.rstrip()]
            f.close()

        if "Attn" in self.args["Prediction"]["name"]:
            self.converter = AttnLabelConverter(self.args["character"], self.device)
        else:
            self.converter = TFMLabelConverter(self.args["character"], self.device)

        self.args["num_class"] = len(self.converter.character)

    def _preprocess(self, image: Image.Image):
        img_tensor = resize(self.resizer, image, self.args)
        return img_tensor

    def _postprocess(self, s: str):
        text_reg = r"(\\(operatorname|mathrm|mathbf|mathsf|mathit|mathfrak|mathnormal)\s?\*? {.*?})"
        letter = "[a-zA-Z]"
        noletter = "[\W_^\d]"
        names = [x[0].replace(" ", "") for x in re.findall(text_reg, s)]
        s = re.sub(text_reg, lambda match: str(names.pop(0)), s)
        news = s

        for space in ["hspace", "vspace"]:
            match = re.finditer(space + " {(.*?)}", news)
            if match:
                new_l = ""
                last = 0
                for m in match:
                    new_l = (
                        new_l + news[last : m.start(1)] + m.group(1).replace(" ", "")
                    )
                    last = m.end(1)
                new_l = new_l + news[last:]
                news = new_l

        return news

    def __call__(self, image: Image.Image, name=None, *arg: Any, **kwargs):
        assert image.mode == "RGB", "input image must be RGB image"
        with torch.no_grad():
            img_tensor = self._preprocess(image).to(self.device)
            text_for_pred = (
                torch.LongTensor(1, self.args["batch_max_length"] + 1)
                .fill_(0)
                .to(self.device)
            )
            preds_index, _, _ = self.model(
                img_tensor, text_for_pred, is_train=False, is_test=True
            )
            pred_str = self.converter.decode(
                preds_index, self.args.get("token_level", "word")
            )[0]

        pred_EOS = pred_str.find("[s]")
        pred_str = pred_str[:pred_EOS]

        process_str = self._postprocess(pred_str)

        return process_str

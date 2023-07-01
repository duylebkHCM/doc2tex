import torch
import torch.nn as nn
from ..module.component.prediction_head import (
    Attention,
    AttentionV2,
    TransformerPrediction,
)


class PredictBuilder(nn.Module):
    def __init__(self, flow, config, SequenceModeling_output):
        super().__init__()
        self.flow = flow
        self.config = config
        if flow["Pred"] == "CTC":
            self.Prediction = nn.Linear(SequenceModeling_output, config["num_class"])

        elif flow["Pred"] == "Attn":
            config["Prediction"]["params"]["num_classes"] = config["num_class"]
            config["Prediction"]["params"]["device"] = config["device"]
            self.Prediction = Attention(**config["Prediction"]["params"])
        elif flow["Pred"] == "Attnv2":
            config["Prediction"]["params"]["num_classes"] = config["num_class"]
            config["Prediction"]["params"]["device"] = config["device"]
            self.Prediction = AttentionV2(**config["Prediction"]["params"])
        elif flow["Pred"] == "Multistage_Attn":
            config["Prediction"]["params"]["num_classes"] = config["num_class"]
            config["Prediction"]["params"]["device"] = config["device"]
            self.Prediction = AttentionV2(**config["Prediction"]["params"])
        elif flow["Pred"] == "TFM":
            config["Prediction"]["params"]["num_classes"] = config["num_class"]
            config["Prediction"]["params"]["device"] = config["device"]
            self.Prediction = TransformerPrediction(**config["Prediction"]["params"])
        else:
            raise ValueError("Prediction name is not suppported")

    def forward(
        self, contextual_feature, text, is_train=True, is_test=False, rtl_text=None
    ):
        beam_size = self.config.get("beam_size", 1)

        addition_outputs = {}
        decoder_attn = None

        if self.flow["Pred"] == "CTC":
            prediction = self.Prediction(contextual_feature.contiguous())

        elif self.flow["Pred"] in ["Attn", "Attnv2"]:
            prediction, logits, decoder_attn = self.Prediction(
                beam_size,
                contextual_feature.contiguous(),
                text,
                is_train=is_train,
                is_test=is_test,
                batch_max_length=self.config["batch_max_length"],
            )

        elif self.flow["Pred"] == "TFM":
            prediction, logits = self.Prediction(
                beam_size, contextual_feature.contiguous(), text, is_test
            )
            self.Prediction.reset_beam()

        return prediction, logits, decoder_attn, addition_outputs

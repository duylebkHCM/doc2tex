import torch.nn as nn
from .build_feat import FeatExtractorBuilder
from .build_seq import SeqModelingBuilder
from .build_pred import PredictBuilder


class Model(nn.Module):
    def __init__(self, opt):
        super(Model, self).__init__()
        self.opt = opt

        stages = {
            "Feat": opt["FeatureExtraction"]["name"],
            "Seq": opt["SequenceModeling"]["name"],
            "Pred": opt["Prediction"]["name"],
        }
        self.stages = stages
        if stages["Seq"].__contains__("Vi"):
            assert stages["Feat"] == "None"

        """ FeatureExtraction """
        self.featextractor = FeatExtractorBuilder(stages, opt)
        FeatureExtraction_output = getattr(
            self.featextractor, "FeatureExtraction_output", None
        )

        """ Sequence modeling"""
        self.seqmodeler = SeqModelingBuilder(stages, opt, FeatureExtraction_output)
        SequenceModeling_output = getattr(
            self.seqmodeler, "SequenceModeling_output", None
        )

        """ Prediction """
        self.predicter = PredictBuilder(stages, opt, SequenceModeling_output)

    def forward_encoder(self, input, *args, **kwargs):
        """Feature extraction stage"""
        visual_feature = self.featextractor(input)
        """ Sequence modeling stage """
        contextual_feature, output_shape, feat_pad = self.seqmodeler(
            visual_feature, *args, **kwargs
        )
        return contextual_feature, output_shape, feat_pad

    def forward_decoder(
        self, contextual_feature, text, is_train=True, is_test=False, rtl_text=None
    ):
        """Prediction stage"""
        prediction, logits, decoder_attn, addition_outputs = self.predicter(
            contextual_feature, text, is_train, is_test, rtl_text
        )

        return prediction, logits, decoder_attn, addition_outputs

    def forward(self, input, text, is_train=True, is_test=False, rtl_text=None):
        contextual_feature, output_shape, feat_pad = self.forward_encoder(input)
        prediction, logits, decoder_attn, addition_outputs = self.forward_decoder(
            contextual_feature,
            text=text,
            is_train=is_train,
            is_test=is_test,
            rtl_text=rtl_text,
        )

        if decoder_attn is not None and output_shape is not None:
            if self.stages["Pred"] == "Attn" and self.stages["Seq"] == "ViT":
                decoder_attn = decoder_attn[:, 1:]
            decoder_attn = decoder_attn.reshape(-1, output_shape[0], output_shape[1])

            addition_outputs.update(
                {
                    "decoder_attn": decoder_attn,
                    "feat_width": output_shape[0],
                    "feat_height": output_shape[1],
                    "feat_pad": feat_pad,
                }
            )

        return prediction, logits, addition_outputs


if __name__ == "__main__":
    import yaml
    import torch

    with open(
        "/media/huynhtruc0309/DATA/Math_Expression/my_source/Math_Recognition/config/train/paper_experiments/report_paper/best_model_noaugment/experiment_1806.yaml",
        "r",
    ) as f:
        config = yaml.safe_load(f)

    config["num_class"] = 499
    config["device"] = "cpu"
    model = Model(config)

    a = torch.rand(1, 1, 32, 224)

    output = model(a)

    print("pred", output[0].shape)

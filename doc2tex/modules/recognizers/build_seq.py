import torch.nn as nn
from modules.component.seq_modeling import BidirectionalLSTM, create_vit_modeling
from modules.component.seq_modeling.bilstm import BiLSTM_Seq_Modeling
from modules.component.common import PositionalEncoding2D, PositionalEncoding1D


class SeqModelingBuilder(nn.Module):
    def __init__(self, flow: dict, config, FeatureExtraction_output):
        super().__init__()
        self.config = config
        self.flow = flow

        if flow["Seq"] == "BiLSTM":
            hidden_size = config["SequenceModeling"]["params"]["hidden_size"]
            use_pos_enc = config["SequenceModeling"]["params"].get("pos_enc", False)

            if use_pos_enc:
                self.image_positional_encoder = PositionalEncoding1D(hidden_size)

            self.SequenceModeling = nn.Sequential(
                BidirectionalLSTM(FeatureExtraction_output, hidden_size, hidden_size),
                BidirectionalLSTM(hidden_size, hidden_size, hidden_size),
            )

            self.SequenceModeling_output = hidden_size

        elif flow["Seq"] == "ViT":
            assert (
                config["max_dimension"] is not None
            ), "ViT encoder require exact height or max height and max width"
            self.SequenceModeling = create_vit_modeling(config)

        else:
            print("No SequenceModeling module specified")
            if flow["Pred"] == "TFM":
                self.image_positional_encoder = PositionalEncoding2D(
                    FeatureExtraction_output
                )

            self.SequenceModeling_output = FeatureExtraction_output

    def forward(self, visual_feature, *args, **kwargs):
        output_shape = None
        pad_info = None

        if self.flow["Seq"] != "BiLSTM":
            contextual_feature = self.SequenceModeling(visual_feature)

            if hasattr(self, "image_positional_encoder"):
                assert len(contextual_feature.shape) == 3
                contextual_feature_1 = self.image_positional_encoder(
                    visual_feature.permute(1, 0, 2)
                )
                contextual_feature_1 = contextual_feature_1.permute(1, 0, 2)
                contextual_feature = self.gated(
                    contextual_feature_1, contextual_feature
                )

        elif self.flow["Seq"] == "ViT":
            contextual_feature, pad_info, feat_shape = self.SequenceModeling(
                visual_feature
            )
            output_shape = (
                feat_shape["height"] // self.SequenceModeling.patch_embed.patch_size[0],
                feat_shape["width"] // self.SequenceModeling.patch_embed.patch_size[1],
            )

        else:
            if self.flow["Pred"] == "TFM":
                contextual_feature = self.image_positional_encoder(visual_feature)
                contextual_feature = contextual_feature.flatten(
                    start_dim=2
                )  # B,C,H,W -> B,C,H*W
                contextual_feature = contextual_feature.transpose(
                    1, 2
                )  # B,C,H*W -> B,H*W,C
            elif self.flow["Pred"] == "Attn":
                contextual_feature = self.AdaptiveAvgPool(
                    visual_feature.permute(0, 3, 1, 2)
                )  # [b, c, h, w] -> [b, w, c, h]
                contextual_feature = contextual_feature.squeeze(3)

        return contextual_feature, output_shape, pad_info

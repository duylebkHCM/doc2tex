import torch.nn as nn
from modules.component.feature_extractor import (
    VGG_FeatureExtractor,
    ResNet_FeatureExtractor,
)


class FeatExtractorBuilder(nn.Module):
    def __init__(self, flow: dict, config):
        super().__init__()
        self.config = config
        self.flow = flow
        self.feat_name = flow["Feat"]

        if self.feat_name != "None":
            mean_height = config["FeatureExtraction"]["params"].pop("mean_height", True)

            if self.feat_name == "VGG":
                self.FeatureExtraction = VGG_FeatureExtractor(
                    **config["FeatureExtraction"]["params"]
                )
                self.FeatureExtraction_output = config["FeatureExtraction"]["params"][
                    "output_channel"
                ]
            elif self.feat_name == "ResNet":
                self.FeatureExtraction = ResNet_FeatureExtractor(
                    **config["FeatureExtraction"]["params"]
                )
                self.FeatureExtraction_output = config["FeatureExtraction"]["params"][
                    "output_channel"
                ]

            if mean_height:
                self.AdaptiveAvgPool = nn.AdaptiveAvgPool2d(
                    (None, 1)
                )  # Transform final (imgH/16-1) -> 1
            else:
                self.proj_feat = nn.Linear(
                    self.FeatureExtraction_output * 3, self.FeatureExtraction_output
                )
        else:
            if flow["Seq"] != "ViT":
                raise Exception("No FeatureExtraction module specified")
            else:
                self.FeatureExtraction = nn.Identity()

    def forward(self, input):
        visual_feature = self.FeatureExtraction(input)

        if self.flow["Seq"] == "BiLSTM":
            if hasattr(self, "AdaptiveAvgPool"):
                visual_feature = self.AdaptiveAvgPool(
                    visual_feature.permute(0, 3, 1, 2)
                )  # [b, c, h, w] -> [b, w, c, 1]
                visual_feature = visual_feature.squeeze(3)
            else:
                visual_feature = visual_feature.permute(0, 3, 1, 2)
                visual_feature = visual_feature.flatten(
                    start_dim=-2
                )  # [b, c, h, w] -> [b, w, c*h]
                visual_feature = self.proj_feat(visual_feature)

        return visual_feature

from .attn_converter import AttnLabelConverter


def create_converter(config, device):
    if "Attn" in config["Prediction"]["name"]:
        converter = AttnLabelConverter(config["character"], device)
    return converter

from .attn_converter import AttnLabelConverter
from .tfm_converter import TFMLabelConverter


def create_converter(config, device):
    with open(config['vocab'], 'r') as f:
        config['character'] = [c.strip() for c in f.readlines()]
    if "Attn" in config["Prediction"]["name"]:
        converter = AttnLabelConverter(config["character"], device)
    elif config["Prediction"]["name"] in ["TFM", "MS_TFM"]:
        converter = TFMLabelConverter(config["character"], device)
    return converter

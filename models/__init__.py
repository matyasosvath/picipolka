import torch

from models import vision_transformer, cnn
from models import puli2, puli3sx, puli_llumix


def get_model(model_name: str, config: dict) -> torch.nn.Module:

    assert isinstance(model_name, str), "Model name must be string!"

    if model_name in ("puli2", "puli3sx", "puli-llumix"):
        model = get_nlp_models(model_name)

    elif model_name in ("cnn", "vit"):
        model = get_vision_models(model_name, config)

    else:
        raise ValueError(f"Model unrecognised! Got {model_name}.")

    print(f"Loaded model: {model_name}")

    return model


def get_nlp_models(model_name: str) -> torch.nn.Module:

    if model_name == "puli2":

        model_args = puli2.ModelArgs()
        model = puli2.Puli2(model_args)

    elif model_name == "puli3sx":

        model_args = puli3sx.ModelArgs()
        model = puli3sx.Puli3SX(model_args)

    elif model_name == "puli-llumix":

        model_args = puli_llumix.ModelArgs()
        model = puli_llumix.PuliLlumix(model_args)

    return model


def get_vision_models(model_name: str, config: dict) -> torch.nn.Module:

    if model_name == "cnn":

        model = cnn.Net()

    elif model_name == "vit":

        model_args = vision_transformer.ModelConfig()

        model_args.n_classes = config["n_classes"]
        model_args.channels =  config["channels"]
        model_args.patch_size = config["patch_size"]
        model_args.img_size =  config["width"]

        model = vision_transformer.ViT(model_args)

    return model

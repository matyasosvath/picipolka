
import cnn
import models.puli2 as puli2
import models.puli3sx as puli3sx
import puli_llumix
import torch
import vision_transformer


def get_model(model_name: str) -> torch.nn.Module:

    assert isinstance(model_name, str), "Model name must be string!"

    if model_name in ("puli2-gpt", "puli3-gpt-neox", "puli-llumix"):
        model = get_nlp_models(model_name)

    elif model_name in ("cnn", "vit"):
        model = get_vision_models(model_name)

    else:
        raise ValueError(f"Model unrecognised! Got {model_name}.")

    print(f"Loaded model: {model_name}")

    return model


def get_nlp_models(model_name: str) -> torch.nn.Module:

    if model_name == "puli2-gpt":

        model_args = puli2.ModelArgs()
        model = puli2.Puli2GPT(model_args)

    elif model_name == "puli3-gpt-neox":

        model_args = puli3sx.ModelArgs()
        model = puli3sx.Puli3GptNeox(model_args)

    elif model_name == "puli-llumix":

        model_args = puli_llumix.ModelArgs()
        model = puli_llumix.PuliLlumix(model_args)

    return model


def get_vision_models(model_name: str) -> torch.nn.Module:

    if model_name == "cnn":

        model = cnn.Net()

    elif model_name == "vit":

        model_args = vision_transformer.ModelConfig()
        model = vision_transformer.ViT(model_args)

    return model

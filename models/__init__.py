from typing import Optional, Tuple

import os
import zipfile
import urllib.request
from tqdm import tqdm

import torch

from models import vision_transformer, cnn
from models import puli2, puli3sx, puli_llumix


_MODELS = {
    "puli2": "https://nc.nlp.nytud.hu/s/cCqHLJaftNnRmGZ/download/puli2-gpt.pt",
}


def get_model(model_name: str, fine_tune: bool, config: dict, compile: bool = True) -> torch.nn.Module:

    assert isinstance(model_name, str), "Model name must be string!"

    if model_name in ("puli2", "puli3sx", "puli-llumix"):
        model = get_nlp_models(model_name, fine_tune)

    elif model_name in ("cnn", "vit"):
        model = get_vision_models(model_name, fine_tune, config)

    else:
        raise ValueError(f"Model unrecognised! Got {model_name}.")

    print(f"Initialized model {model_name}")

    if fine_tune:
        model_path = _download_artifact(model_name)
        model.load_state_dict(torch.load(f=model_path, weights_only=True))
        print(f"Loaded trained model {model_name}")

    # if compile:
    #     model = torch.compile(model, mode="reduce-overhead", fullgraph=True)

    return model


def get_nlp_models(model_name: str, fine_tune: bool) -> torch.nn.Module:

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


def get_vision_models(model_name: str, fine_tune: bool, config: dict) -> torch.nn.Module:

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


def _download_artifact(model_name: str) -> str:

    if model_name not in _MODELS:
        raise RuntimeError(f"Model {model_name} not found; available models: {_MODELS}")

    default = os.path.join(os.path.expanduser("~"), ".cache")
    artifact_path = os.path.join(os.getenv("XDG_CACHE_HOME", default), f"puli/{model_name}")

    model_url = _MODELS[model_name]

    if os.path.isdir(artifact_path):
        print(f"Model path for {model_name} already exists! Skipping download.")
    else:
        _download(model_url, artifact_path)

    model_file_path = artifact_path + f"/{model_name}.pt"

    return model_file_path


def _download(url: str, target_dir: str, unzip: bool = False) -> None:

    os.makedirs(target_dir, exist_ok=True)

    file_name = os.path.basename(url)
    download_path = os.path.join(target_dir, file_name)

    with urllib.request.urlopen(url) as source, open(download_path, "wb") as output:
        with tqdm(
            desc="Downloading data",
            total=int(source.info().get("Content-Length")),
            ncols=80,
            unit="iB",
            unit_scale=True,
            unit_divisor=1024,
        ) as loop:
            while True:
                buffer = source.read(8192)
                if not buffer:
                    break

                output.write(buffer)
                loop.update(len(buffer))

    if unzip:
        _unzip_file(download_path, target_dir)
        os.remove(download_path)


def _unzip_file(download_path: str, target_dir: str) -> None:

    with zipfile.ZipFile(download_path, "r") as zip_ref:

        zip_info_list = zip_ref.infolist()
        total_files = len(zip_info_list)

        with tqdm(desc="Unzipping data", total=total_files, unit='file', ncols=80) as progress_bar:
            for zip_info in zip_info_list:
                zip_ref.extract(zip_info, target_dir)
                progress_bar.update(1)

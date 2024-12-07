from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class ModelConfig: # tiny-ViT
    hidden_size: int = 192
    n_heads: int = 3
    n_layers: int = 3
    mlp_size: int = 768
    attn_dropout: float = 0.0
    mlp_dropout: float = 0.1
    qkv_bias: bool = False

    # defined later
    n_classes: int = -1
    channels: int = -1
    patch_size: int = -1
    img_size: int = -1


class PatchEmbedding(nn.Module):

    def __init__(self, model_config: ModelConfig) -> None:
        super().__init__()

        self.patcher = nn.Conv2d(
            in_channels= model_config.channels,
            out_channels = model_config.hidden_size,
            kernel_size=model_config.patch_size,
            stride=model_config.patch_size,
            padding=0
        )

        self.flatten = nn.Flatten(
            start_dim=2, # only flatten the feature map dimensions into a single vector
            end_dim=3
        )

        self.patch_size = model_config.patch_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        # (batch_size, channel, height, width)
        image_resolution = x.shape[-1]

        assert image_resolution % self.patch_size == 0, f"Input image size must be divisble by patch size."

        # (batch_size, channel, height, width) -> (batch_size, hidden_size, height / patch_size, width / patch_size)
        # hidden_size = patch_size * patch_size * channels
        x_patched = self.patcher(x)

        # (batch_size, hidden_size, height / patch_size, width / patch_size) -> (batch_size, hidden_size, height / patch_size * width / patch_size)
        # patches_num = height / patch_size * width / patch_size
        x_flattened = self.flatten(x_patched)

        # also called seq length
        # (batch_size, hidden_size, patches_num x patches_num) -> (batch_size, patches_num x patches_num, hidden_size),
        x_permuted = x_flattened.permute(0, 2, 1)

        return x_permuted

class MultiHeadAttention(nn.Module):

    def __init__(self, model_config: ModelConfig) -> None:
        super().__init__()

        self.ln = nn.LayerNorm(model_config.hidden_size)

        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=model_config.hidden_size,
            num_heads=model_config.n_heads,
            dropout=model_config.attn_dropout,
            batch_first=True
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.ln(x)
        attn_output, _ = self.multihead_attn(query=x, key=x, value=x, need_weights=False)
        return attn_output

class MLP(nn.Module):
    def __init__(self, model_config: ModelConfig) -> None:
        super().__init__()

        self.ln = nn.LayerNorm(model_config.hidden_size)

        self.mlp = nn.Sequential(
            nn.Linear(in_features = model_config.hidden_size, out_features = model_config.mlp_size),
            nn.GELU(),
            nn.Dropout(model_config.mlp_dropout),
            nn.Linear(in_features = model_config.mlp_size, out_features = model_config.hidden_size),
            nn.Dropout(model_config.mlp_dropout)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.ln(x)
        x = self.mlp(x)
        return x

class Encoder(nn.Module):
    def __init__(self, model_config: ModelConfig) -> None:
        super().__init__()

        self.msa = MultiHeadAttention(model_config)
        self.mlp =  MLP(model_config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x =  self.msa(x) + x
        x = self.mlp(x) + x
        return x

class ViT(nn.Module):
    def __init__(self, model_config: ModelConfig) -> None:
        super().__init__()

        assert model_config.img_size % model_config.patch_size == 0, f"Image size must be divisible by patch size."

        self.num_patches = (model_config.img_size * model_config.img_size) // model_config.patch_size**2

        self.class_embedding = nn.Parameter(data=torch.randn(1, 1, model_config.hidden_size), requires_grad=True)

        self.position_embedding = nn.Parameter(data=torch.randn(1, self.num_patches+1, model_config.hidden_size), requires_grad=True)

        self.patch_embedding = PatchEmbedding(model_config)

        self.transformer_encoder = nn.Sequential(
            *[
                Encoder(model_config)
                for _ in range(model_config.n_layers)
            ]
        )

        self.classifier = nn.Sequential(
            nn.LayerNorm(model_config.hidden_size),
            nn.Linear(model_config.hidden_size, model_config.n_classes)
        )

    def forward(self, x):

        batch_size = x.shape[0]

        # expand to match the batch size, "-1" infer dimension
        class_token = self.class_embedding.expand(batch_size, -1, -1)

        x = self.patch_embedding(x)
        x = torch.cat((class_token, x), dim=1)
        x = self.position_embedding + x

        x = self.transformer_encoder(x)

        x = self.classifier(x[:, 0])

        return x

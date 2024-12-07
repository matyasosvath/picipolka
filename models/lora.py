import math
import torch
from torch import nn


class LoRALayer(torch.nn.Module):

    def __init__(self, in_dim: int, out_dim: int, rank: int, alpha: float) -> None:
        super().__init__()

        self.A = torch.nn.Parameter(torch.empty(in_dim, rank))
        torch.nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        self.B = torch.nn.Parameter(torch.zeros(rank, out_dim))
        self.alpha = alpha

    def forward(self, x):
        x = self.alpha * (x @ self.A @ self.B)
        return x


class LinearLayerWithLoRA(torch.nn.Module):

    def __init__(self, in_features: int, out_features: int, rank: int, alpha: float) -> None:
        super().__init__()

        self.linear =  nn.Linear(in_features, out_features)
        self.lora = LoRALayer(in_features, out_features, rank, alpha)

    def forward(self, x)-> torch.Tensor:
        return self.linear(x) + self.lora(x)
import torch
from torch import nn


class KDHead(nn.Module):
    def __init__(self, input_dimension: int, output_dimension: int):
        super(KDHead, self).__init__()
        # Output shape is teacher shape
        self.projection = nn.Linear(input_dimension, output_dimension)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.normalize(self.projection(x), p=2, dim=-1)

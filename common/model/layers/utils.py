import torch
from einops import rearrange
from torch import nn


class TimeStepsCompression(nn.Module):
    def forward(self, x: torch.Tensor):
        return rearrange(x, 'b T F D -> (b T) F D')


class TimeStepsDecompression(nn.Module):
    def forward(self, x: torch.Tensor, b: int):
        return rearrange(x, '(b T) F D -> b T F D', b=b)

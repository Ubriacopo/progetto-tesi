import torch
from torch import nn

from main.utils.data import MaskedValue


class TimedMaskedAdapter(nn.Module):
    pass


class AudioAdapter(nn.Module):
    def __init__(self, input_size: int, project_out_size: int = None):
        super().__init__()
        self.ff = nn.Sequential(
            nn.Linear(input_size, input_size * 4),
            nn.GELU(),
            nn.LayerNorm(input_size * 4),
            nn.Linear(input_size * 4, project_out_size),
        )

    def forward(self, x: torch.Tensor, mask=None):
        """
        problema quii
        :param x: [b T P D]
        :param mask: [b T]
        :return:
        """
        y = self.ff(x)
        return MaskedValue(data=y, mask=mask)

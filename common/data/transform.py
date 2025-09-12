from abc import ABC, abstractmethod
from typing import Optional

import torch
from torch import nn
from torchvision.transforms import Lambda

IDENTITY = Lambda(lambda x: x)


class SequenceResampler(nn.Module, ABC):
    def __init__(self, sequence_length: int, resampler: nn.Module = IDENTITY):
        super(SequenceResampler, self).__init__()
        self.sequence_length = sequence_length
        self.resampler = resampler

    @abstractmethod
    def get_sequence_dim_value(self, x: torch.Tensor) -> int:
        raise NotImplementedError("SequenceResampler.get_sequence_dim_value")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        T = self.get_sequence_dim_value(x)
        segments = int(T / self.sequence_length)
        if T % self.sequence_length != 0:
            segments += 1  # We always have a segment if not exact division.

        y: Optional[torch.Tensor] = None

        for i in range(int(segments)):
            x_i = x[i * self.sequence_length:(i + 1) * self.sequence_length]
            res = self.resampler(x_i)
            # We have new dimension that records the sequence.
            res = res.unsqueeze(0)

            y: torch.Tensor = torch.cat((y, res)) if y is not None else res

        return y

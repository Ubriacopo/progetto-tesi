from abc import ABC, abstractmethod
from typing import Optional, Literal

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


class MultimediaPadding(nn.Module):
    def __init__(self, max_length: int, drop_mask: bool = False, padding: Literal['zero', 'none', 'last'] = 'zero'):
        super(MultimediaPadding, self).__init__()
        self.max_length = max_length
        self.drop_mask = drop_mask
        self.padding: Literal['zero', 'last', 'none'] = padding

    def forward(self, x: torch.Tensor) -> dict | torch.Tensor:
        # Base case
        T = x.shape[0]
        if T > self.max_length:
            raise ValueError("We suppose to have a max sequence length during sampling."
                             "A larger value is therefore impossible")

        # The input x is okay so we can just return it.
        if T == self.max_length:
            mask = torch.ones(T).bool()
            return {"data": x, "mask": mask} if not self.drop_mask else x

        # We have to pad
        if self.padding == 'zero':
            pad = torch.zeros_like(x[0]).unsqueeze(0).repeat_interleave(self.max_length - T, dim=0)
            x = torch.cat([x, pad])
            mask = torch.zeros(self.max_length).bool()
            mask[:T] = True
            return {"data": x, "mask": mask} if not self.drop_mask else x

        raise ValueError("Given padding strategy is not supported.")

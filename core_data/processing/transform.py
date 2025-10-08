import dataclasses
from abc import ABC, abstractmethod
from typing import Optional, Literal

import torch
from torch import nn
from torchvision.transforms import Lambda

from core_data.data_point import FlexibleDatasetPoint
from core_data.media.media import Media
from model.utils import MaskedResult

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


class Branch(nn.Module):
    def __init__(self, *modules: tuple[str, nn.Module], as_dict: bool = True):
        super().__init__()
        self.branches = nn.ModuleList(modules[:][1])
        self.keys = modules[:][0]
        # Output a dictionary or a tuple.
        self.as_dict = as_dict

    def forward(self, x):
        outs = [m(x) for m in self.branches]
        if self.as_dict:
            if self.keys is None or len(self.keys) != len(outs):
                raise ValueError("keys must match number of modules")
            return {k: v for k, v in zip(self.keys, outs)}
        return tuple(outs)


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
                             f"A larger value is therefore impossible. ({T} > {self.max_length})")

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


class ReplaceMedia(nn.Module):
    # noinspection PyMethodMayBeStatic
    def forward(self, x: Media):
        return dataclasses.replace(x)


class SegmentsCalculationTransform(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass


class ProcessSegments(nn.Module):
    def __init__(self, *processing: nn.Module):
        super().__init__()
        self.net = nn.Sequential(*processing)

    def forward(self, x: FlexibleDatasetPoint, segments: list[tuple[int, int]]):
        res = []
        for interval in segments:
            x.interval = interval
            res.append(self.net(x))
        return self.net(x)


class ToSimpleMaskedObject(nn.Module):
    def __init__(self, stop_at_dim: Optional[int] = -1):
        super().__init__()
        self.stop_at_dim: Optional[int] = stop_at_dim

    def forward(self, x: torch.Tensor) -> MaskedResult:
        # Drop masking on last dimension
        shape = x.shape[:self.stop_at_dim] if self.stop_at_dim is not None else x.shape
        return {"data": x, "mask": torch.ones(shape)}
import dataclasses

import torch

from main.utils.data import MaskedValue


@dataclasses.dataclass
class EegBaseModelOutputs:
    embeddings: torch.Tensor
    kd_outs: dict[str, MaskedValue]
    multimodal_outs: dict[str, MaskedValue]


@dataclasses.dataclass
class WeaklySupervisedEegBaseModelOutputs(EegBaseModelOutputs):
    pred: torch.Tensor

import dataclasses

import torch

from main.utils.data import MaskedValue


@dataclasses.dataclass
class EegBaseModelOutputs:
    cls: torch.Tensor
    embeddings: MaskedValue
    kd_outs: dict[str, MaskedValue]
    multimodal_outs: dict[str, MaskedValue]


@dataclasses.dataclass
class WeaklySupervisedEegBaseModelOutputs(EegBaseModelOutputs):
    pred: torch.Tensor

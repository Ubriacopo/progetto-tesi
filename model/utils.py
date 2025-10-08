from typing import TypedDict, Optional

import torch


def freeze_module(m: torch.nn.Module):
    for p in m.parameters():
        p.requires_grad = False


class MaskedResult(TypedDict):
    data: torch.Tensor
    mask: Optional[torch.Tensor]

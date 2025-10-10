from typing import Iterable

import torch
import torch.nn.functional as F


@torch.no_grad()
def temporal_pyramid_pooling_3d(x: torch.Tensor, levels: Iterable[int] = (1, 2, 4, 8, 17)):  # x: (P, D)
    # Change shape from [T, P, D] -> [T, D, P] for pooling. (temporal positions / tokens are pooled thus temporal pooling).
    outs = [F.adaptive_avg_pool1d(x.transpose(1, 2), L).transpose(1, 2) for L in levels]  # [(T, L, D)]
    return torch.cat(outs, dim=1)  # (T, sum(levels)=32, D))

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


def top_k_hits_from_sim(sim: torch.Tensor, ks: tuple[int, ...]) -> dict[int, torch.Tensor]:
    """

    :param sim: Similarity matrix of size (N, M)
    :param ks:
    :return:
    """
    gt = torch.arange(sim.size(0), device=sim.device)
    out = {}
    kmax = min(max(ks), sim.size(1))
    top = sim.topk(kmax, dim=1).indices  # (n, kmax)

    # Compare once
    eq = top.eq(gt[:, None])  # (n, kmax)
    for k in ks:
        k = min(k, sim.size(1))
        out[k] = eq[:, :k].any(dim=1).float().mean()

    return out

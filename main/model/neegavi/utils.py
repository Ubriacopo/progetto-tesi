import dataclasses

import torch

from main.utils.data import MaskedValue


# todo ridurre per inference
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


import torch


def retrieval_metrics_chunked(
        f: torch.Tensor,
        e: torch.Tensor,
        top_k=(1, 3, 5, 10),
        chunk_size: int = 256,
):
    """
    Exact chunked retrieval metrics for paired embeddings.
    Assumes row i matches column i.
    Similarity: dot product.
    """
    assert f.ndim == 2 and e.ndim == 2
    assert f.shape == e.shape

    device = f.device
    n = f.size(0)
    cols = torch.arange(n, device=device)

    # positive similarities = diagonal of f @ e.T

    rr_sum = torch.zeros((), device=device, dtype=torch.float32)
    recall_sums = {k: torch.zeros((), device=device, dtype=torch.float32) for k in top_k}
    sim_diag_sum = torch.zeros((), device=device, dtype=torch.float32)
    margin_sum = torch.zeros((), device=device, dtype=torch.float32)

    for s in range(0, n, chunk_size):
        t = min(s + chunk_size, n)

        f_chunk = f[s:t]  # (B, D)
        sim = f_chunk @ e.T  # (B, N)

        rows = torch.arange(s, t, device=device)  # global row ids
        local_rows = torch.arange(t - s, device=device)  # local row ids within chunk

        pos = sim[local_rows, rows]  # exact diagonal entries from sim
        pos_col = pos[:, None]  # (B, 1)

        # rank with exact tie handling
        greater = (sim > pos_col).sum(dim=1)

        equal = (sim == pos_col)
        rows = torch.arange(s, t, device=device)
        tie_before = (equal & (cols[None, :] < rows[:, None])).sum(dim=1)

        rank = greater + tie_before + 1  # 1-based rank

        # MRR
        rr_sum += (1.0 / rank.to(torch.float32)).sum()

        # Recall@K
        for k in top_k:
            recall_sums[k] += (rank <= k).to(torch.float32).sum()

        # Alignment = mean diagonal similarity
        sim_diag_sum += pos.to(torch.float32).sum()

        # Margin = mean(pos - mean_neg)
        row_sum = sim.sum(dim=1)
        if n > 1:
            neg_mean = (row_sum - pos) / (n - 1)
            margin_sum += (pos - neg_mean).to(torch.float32).sum()
        else:
            margin_sum += torch.zeros((), device=device, dtype=torch.float32)

    mrr = rr_sum / n
    recalls = {k: recall_sums[k] / n for k in top_k}
    mean_r = torch.stack([recalls[k] for k in top_k]).mean()
    alignment = sim_diag_sum / n
    margin = margin_sum / n

    return {"mrr": mrr, "recalls": recalls, "mean_r": mean_r, "alignment": alignment, "margin": margin, }


def get_model_ckpt(weights_path: str, check_path: str = "student.", is_finetune: bool = False):
    ckpt = torch.load(weights_path, map_location="cpu")
    my_ckpt = dict()

    for key, value in ckpt["state_dict"].items():
        if is_finetune:
            key = key.replace("pivot.adapter.", "pivot.adapter.adapter.", 1)
        if key.startswith(check_path):
            my_ckpt[key[len(check_path):]] = value

    return my_ckpt


def get_model_ckpt_finetune(weights_path: str, check_path: str = "student."):
    ckpt = torch.load(weights_path, map_location="cpu")
    my_ckpt = dict()
    for key, value in ckpt["state_dict"].items():
        if key.startswith(check_path):
            key = key.replace("pivot.adapter.", "pivot.adapter.adapter.", 1)
            my_ckpt[key[len(check_path):]] = value

    return my_ckpt

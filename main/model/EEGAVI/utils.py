import torch
from torch.nn import functional as F


# todo explain and review
def interval_overlap_weights(t_source: int, t_target: int, device=None):
    """
    Returns W of shape (T_tgt, T_src) with non-negative rows summing to 1
    that describe how much of each source interval contributes to each target interval.

    Intervals are uniform partitions of [0, 1].
    """
    device = device or "cpu"

    # Interval edges
    src_edges = torch.linspace(0, 1, t_source + 1, device=device)
    tgt_edges = torch.linspace(0, 1, t_target + 1, device=device)
    # Interval spans
    src_st, src_en = src_edges[:-1], src_edges[1:]
    tgt_st, tgt_en = tgt_edges[:-1], tgt_edges[1:]

    # Compute pairwise overlaps
    # overlap(t,s) = max(0, min(t_en, s_en) - max(t_st, s_st))
    t_st = tgt_st[:, None]  # (t_target, 1)
    t_en = tgt_en[:, None]  # (t_target, 1)
    s_st = src_st[None, :]  # (1, t_source)
    s_en = src_en[None, :]  # (1, t_source)

    overlap = (torch.minimum(t_en, s_en) - torch.maximum(t_st, s_st)).clamp(min=0)  # (T_tgt, T_src)
    # Normalize rows to sum to 1 where possible
    row_sum = overlap.sum(dim=1, keepdim=True)
    w = torch.where(row_sum > 0, overlap / row_sum, overlap)  # rows with all-zero stay zero
    return w  # (T_tgt, T_src)


# todo revisiona
def remap_with_overlap(x: torch.Tensor, mask: torch.Tensor, t: int):
    """

    :param x: (b, T, P, D)
    :param mask: (b, T)
    :param t: Is T' (new T to map to)
    :return:
        (b, T', P, D)  remapped x
        (b, T')        remapped mask
    """
    keep_p = x.dim() == 4
    if not keep_p:  # upgrade to a (P) axis for unified math
        x = x.unsqueeze(2)  # (B, T_src, 1, D)
    b, T, P, D = x.shape
    w = interval_overlap_weights(t_source=T, t_target=t, device=x.device)
    # Mask invalid source steps
    bmask = mask.bool()
    wb = w[None, :, :] * bmask[:, None, :].to(x.dtype)  # (B, T_tgt, T_src)
    # Denominator per target bin (how much valid mass contributed)
    valid_mass = wb.sum(dim=2)
    y_mask = valid_mass > 0

    denominator = valid_mass.unsqueeze(-1).clamp(min=1e-9)  # (B, T_tgt, 1)

    # Flatten (P,D) to PD and use bmm: (B, T_tgt, T_src) @ (B, T_src, PD) -> (B, T_tgt, PD)
    x_flat = x.reshape(b, T, P * D).to(wb.dtype)  # ensure dtype matches for matmul
    y_flat = torch.bmm(wb, x_flat) / denominator  # (B, T_tgt, PD)
    y = y_flat.view(b, t, P, D)  # (B, T_tgt, P, D)
    # Target mask is simply whether any valid mass arrived
    if not keep_p:
        y = y.squeeze(2)  # (B, T_tgt, D)

    return y, y_mask


@torch.no_grad()
def batch_stats_generic(X: torch.Tensor,
                        mask: torch.Tensor | None = None,
                        reduce_axes: tuple[int, ...] = (1,),  # e.g. (1,2,3) or (1,2)
                        ):
    """
    X: [B, *A, D]   (any number of reduce axes A, then D)
    mask: [B, *A]   (True=valid) or None
    """
    B = X.size(0)
    X = X.float()

    if mask is not None:
        # bring mask to same rank as X by adding the last dim
        while mask.dim() < X.dim() - 1:
            mask = mask.unsqueeze(-1)
        w = mask.to(X.dtype)

        num = (X * w).sum(dim=reduce_axes)
        den = w.sum(dim=reduce_axes).clamp_min(1e-6)
        Xb = num / den  # [B,D]
    else:
        Xb = X.mean(dim=reduce_axes)  # [B,D]

    Xn = F.normalize(Xb, dim=-1)
    S = Xn @ Xn.T  # [B,B]

    # rank-1 dominance
    Xm = Xn - Xn.mean(0, keepdim=True)
    s = torch.linalg.svdvals(Xm)
    rank1_ratio = (s[0] / (s.sum() + 1e-9)).item()

    Bf = float(B)
    diag = float(S.diag().mean())
    off = float((S.sum() - S.diag().sum()) / (Bf * Bf - Bf))
    return dict(diag=diag, off=off, gap=diag - off,
                S_min=float(S.min()), S_max=float(S.max()),
                rank1_ratio=rank1_ratio,
                across_batch_std=float(Xb.std(0).mean()))


@torch.no_grad()
def batch_stats_5d(x: torch.Tensor, mask: torch.Tensor | None = None, reduce_axes=(1, 2, 3), max_dim=3):
    """
    Pool over T,F,P by default

    X:    [B, T, F, P, D]
    mask: [B, T, F] or [B, T, F, P] (True = valid)
    """
    B = x.size(0)
    x = x.to(torch.float32)

    if mask is not None:
        # broadcast mask to [B,T,F,P,1]
        if mask.dim() == max_dim:  # [B,T,F]
            mask = mask.unsqueeze(-1)  # [B,T,F,1]
        mask = mask.to(x.device)
        w = mask.unsqueeze(-1).to(x.dtype)  # [B,T,F,P,1]

        # masked numerator/denominator
        num = (x * w).sum(dim=reduce_axes)  # -> [B, D]
        den = w.sum(dim=reduce_axes)  # -> [B, 1]
        den = den.clamp_min(1e-6)
        Xb = num / den  # [B, D]
    else:
        Xb = x.mean(dim=reduce_axes)  # [B, D]

    # similarity stats across the batch
    Xn = F.normalize(Xb, dim=-1)
    S = Xn @ Xn.T  # [B, B]
    diag = S.diag().mean().item()
    off = (S.sum() - S.diag().sum()) / (S.numel() - B)

    # rank-1 dominance check
    Xm = Xn - Xn.mean(0, keepdim=True)
    s = torch.linalg.svdvals(Xm)  # faster than full svd
    rank1_ratio = (s[0] / (s.sum() + 1e-9)).item()

    return dict(
        diag=float(diag),
        off=float(off),
        gap=float(diag - off),
        S_min=float(S.min()),
        S_max=float(S.max()),
        rank1_ratio=rank1_ratio,
        across_batch_std=float(Xb.std(0).mean())
    )

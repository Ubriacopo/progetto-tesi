import torch

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
    denominator = wb.sum(dim=2, keepdim=True).clamp(min=1e-9)  # (B, T_tgt, 1)

    # Flatten (P,D) to PD and use bmm: (B, T_tgt, T_src) @ (B, T_src, PD) -> (B, T_tgt, PD)
    x_flat = x.reshape(b, T, P * D).to(wb.dtype)  # ensure dtype matches for matmul
    y_flat = torch.bmm(wb, x_flat) / denominator  # (B, T_tgt, PD)
    y = y_flat.view(b, t, P, D)  # (B, T_tgt, P, D)
    # Target mask is simply whether any valid mass arrived
    y_mask = (denominator.squeeze(-1) > 0)  # (B, T_tgt)
    if not keep_p:
        y = y.squeeze(2)  # (B, T_tgt, D)

    return y, y_mask

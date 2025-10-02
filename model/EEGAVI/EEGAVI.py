from typing import Optional

import torch
from einops import rearrange
from torch import nn

from model.layer.attention.x_attention import GatedXAttentionBlock
from model.layer.base import ModalContextEncoder
from model.layer.modality_stream import ModalityStream
from model.EEGAVI.transforms import media_locs_single_item

import lightning as L


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

    # Weighted sum over source time
    # Expand Wb to broadcast over P,D
    wb_exp = wb[:, :, :, None, None]  # (b, T', T, 1, 1)
    x_expanded = x[:, None, :, :, :]  # (b, 1, T, P, D)
    denominator = denominator.squeeze(-1)
    y = (wb_exp * x_expanded).sum(dim=2) / denominator[:, :, None, None]  # (B, T_tgt, P, D)
    # Target mask: bin valid if it received any valid mass
    y_mask = (bmask[:, None, :] & (w[None] > 0)).any(dim=2)

    if not keep_p:
        y = y.squeeze(2)  # (B, T_tgt, D)
    return y, y_mask


class EEGAVI(L.LightningModule):
    def __init__(self,
                 pivot_latent_size: int, pivot_modality: ModalityStream,
                 supporting_latent_size: int, supporting_modalities: list[ModalityStream], use_modality_encoder: bool,
                 xattn_blocks: int, final_projector: nn.Module,
                 remap_timesteps: int, drop_p: float = 0.1):
        super(EEGAVI, self).__init__()

        self.pivot_modality = pivot_modality
        self.supporting_modalities = nn.ModuleList(supporting_modalities)

        self.modality_encoder: Optional[ModalContextEncoder] = None
        if use_modality_encoder:
            modality_mappings = {e.get_code(): i for i, e in enumerate(supporting_modalities)}
            self.modality_encoder = ModalContextEncoder(supporting_latent_size, modality_mappings)

        self.gatedXAttn_layers = nn.ModuleList([
            GatedXAttentionBlock(dim=pivot_latent_size, dim_latent=supporting_latent_size) for _ in range(xattn_blocks)
        ])

        self.projector: nn.Module = final_projector

        self.remap_timesteps: int = remap_timesteps
        self.drop_p: float = drop_p

    def reshape_to_fixed_timesteps(self, supports: list[torch.Tensor], masks: list[torch.Tensor]):
        return_supports, return_masks = [], []
        for support, mask in zip(supports, masks):
            support, mask = remap_with_overlap(support, mask, self.remap_timesteps)
            support = support * mask[:, :, None, None]

            p = support.shape[2]

            mask = mask[:, :, None].expand(-1, -1, p)
            return_supports.append(support), return_masks.append(mask)

        return return_supports, return_masks

    @staticmethod
    def remask(supp: torch.Tensor, device):
        b, T, F, D = supp.shape
        key_time_idx = torch.arange(T, device=device).repeat_interleave(F)
        # allow[q_t, k] = (time(k) <= q_t)
        allow = key_time_idx.view(1, 1, -1) <= torch.arange(T, device=device).view(1, T, 1)
        return allow

    def forward(self, x: dict, use_kd: bool = False):
        use_kd = use_kd or ("kd" in x and x["kd"])

        kd_outs: dict = {}
        # First work with the base modality. (EEG in our case)
        key: str = self.pivot_modality.get_code()
        base, base_mask = x[key]["data"], x[key]["mask"] if "mask" in x[key] else None
        base = self.pivot_modality(base, mask=base_mask, use_kd=use_kd)
        if isinstance(base, tuple):
            # Store the KD output to return later
            kd_outs[key] = base[1]
            # Now we can really get the resampled embeddings
            base = base[0]

        adapted_supports: list[torch.Tensor] = []
        adapted_supports_masks: list[torch.Tensor] = []

        n_modalities = len(self.supporting_modalities)
        b = next(iter(x.values()))["data"].shape[0]
        if (not self.training) or self.drop_p <= 0:
            keep = torch.ones(b, n_modalities, dtype=torch.bool, device=self.device)
        else:
            keep = torch.bernoulli(torch.full((b, n_modalities), 1 - self.drop_p, device=self.device)).bool()
            dead = ~keep.any(1)
            if dead.any(): keep[dead, torch.randint(0, n_modalities, (dead.sum(),), device=base.device)] = True

        for m, adapter in enumerate(self.supporting_modalities):
            key: str = adapter.get_code()
            # Kept samples
            idx = keep[:, m].nonzero(as_tuple=True)[0]
            if idx.numel() == 0:
                continue  # If None taken we skip entirely

            supp, mask = x[key]["data"], x[key].get("mask", None)
            adapted_supp = adapter(supp[idx], mask=mask[idx] if mask is not None else None, use_kd=use_kd)
            if isinstance(adapted_supp, tuple):
                # Store the KD output to return later
                kd_outs[key] = adapted_supp[1]
                # Now we can really get the embeddings
                adapted_supp = adapted_supp[0]

            # Modality embedding if wanted
            if self.modality_encoder is not None:
                adapted_supp = self.modality_encoder(adapted_supp, modality=key)

            Y = adapted_supp.new_zeros(b, *adapted_supp.shape[1:])
            Y[idx] = adapted_supp

            if mask is None:
                n_modalities = torch.zeros(b, Y.size(1), dtype=torch.bool, device=Y.device)
                n_modalities[idx] = True
            else:
                n_modalities = mask.new_zeros(b, Y.size(1))
                n_modalities[idx] = mask[idx]

            adapted_supports_masks.append(n_modalities)
            adapted_supports.append(Y)

        supports, masks = self.reshape_to_fixed_timesteps(adapted_supports, adapted_supports_masks)
        supp = torch.cat(supports, dim=2)
        supp_mask = torch.cat(masks, dim=2)
        # Prepare attention mask (What is seeable)
        allow = self.remask(supp=supp, device=supp.device)

        # Initialize the variable to anything
        z: torch.Tensor = base
        for gated_x_attn in self.gatedXAttn_layers:
            z = gated_x_attn(z, supp, attn_mask=allow, q_mask=base_mask, kv_mask=supp_mask)

        logits = self.projector(z)
        return (logits, kd_outs) if use_kd else logits

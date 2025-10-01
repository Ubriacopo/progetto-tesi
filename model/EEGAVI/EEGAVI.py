from typing import Optional

import torch
from einops import rearrange
from torch import nn

from model.layer.attention.x_attention import GatedCrossAttentionBlock
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
    wb = w[None, :, :] * mask[:, None, :].to(x.dtype)  # (B, T_tgt, T_src)
    # Denominator per target bin (how much valid mass contributed)
    denominator = wb.sum(dim=2, keepdim=True).clamp(min=1e-9)  # (B, T_tgt, 1)

    # Weighted sum over source time
    # Expand Wb to broadcast over P,D
    wb_exp = wb[:, :, :, None, None]  # (b, T', T, 1, 1)
    x_expanded = x[:, None, :, :, :]  # (b, 1, T, P, D)
    denominator = denominator.squeeze(-1)
    y = (wb_exp * x_expanded).sum(dim=2) / denominator[:, :, None, None]  # (B, T_tgt, P, D)
    # Target mask: bin valid if it received any valid mass
    y_mask = (wb.sum(dim=2) > 0)  # (B, T_tgt)

    if not keep_p:
        y = y.squeeze(2)  # (B, T_tgt, D)
    return y, y_mask


class EEGAVI(L.LightningModule):
    def __init__(self,
                 # EEG
                 pivot_latent_size: int,
                 pivot_modality: ModalityStream,

                 supporting_latent_size: int,
                 supporting_modalities: list[ModalityStream],
                 use_modality_encoder: bool,

                 cross_attention_blocks: int,

                 final_projector: nn.Module,
                 ):
        super(EEGAVI, self).__init__()

        self.pivot_modality = pivot_modality
        self.supporting_modalities = nn.ModuleList(supporting_modalities)

        self.modality_encoder: Optional[ModalContextEncoder] = None
        if use_modality_encoder:
            modality_mappings = {e.get_code(): i for i, e in enumerate(supporting_modalities)}
            self.modality_encoder = ModalContextEncoder(supporting_latent_size, modality_mappings)
        # TODO: random disabler for each supporting modality for training robustness
        # What about modality gating? Cross Attention handles it!
        self.gatedXAttn_layers = nn.ModuleList([
            GatedCrossAttentionBlock(dim=pivot_latent_size, dim_latent=supporting_latent_size)
            for _ in range(cross_attention_blocks)
        ])
        self.projector = final_projector

    def forward(self, x: dict):
        use_kd = "kd" in x and x["kd"]
        # Base Modality first
        kd_outputs: dict = {}

        z_supports: list[torch.Tensor] = []
        z_masks: list[torch.Tensor] = []

        base, base_mask = x[self.pivot_modality.get_code()]['data'], x[self.pivot_modality.get_code()]['mask']
        z_base = self.pivot_modality(base, mask=base_mask, use_kd=use_kd)
        if isinstance(z_base, tuple):
            kd_outputs[self.pivot_modality.get_code()] = z_base[1]
            z_base = z_base[0]

        for adapter in self.supporting_modalities:
            key = adapter.get_code()
            supp, mask = x[key]["data"], x[key]["mask"] if "mask" in x[key] else None
            z_supp = adapter(supp, mask=mask, use_kd=use_kd)

            if isinstance(z_supp, tuple):
                # Store the KD output to return later
                kd_outputs[key] = z_supp[1]
                # Now we can really get the embeddings
                z_supp = z_supp[0]

            if self.modality_encoder is not None:
                z_supp = self.modality_encoder(z_supp, modality=key)

            z_masks.append(mask)
            z_supports.append(z_supp)

        supports: list[torch.Tensor] = []
        masks: list[torch.Tensor] = []
        for z, mask in zip(z_supports, z_masks):
            z, mask = remap_with_overlap(z, mask, 32)  # todo pass
            z = z * mask[:, :, None, None]

            p = z.shape[2]
            mask = mask[:, :, None].expand(-1, -1, p)
            supports.append(z), masks.append(mask)

        z_supp = torch.cat(supports, dim=2)
        z_mask = torch.cat(masks, dim=2)

        # Prepare attention masks / masks in general
        b, T, F, D = z_supp.shape
        key_time_idx = torch.arange(T, device=z_base.device).repeat_interleave(F)
        # allow[q_t, k] = (time(k) <= q_t)
        allow = key_time_idx.view(1, 1, -1) <= torch.arange(T, device=z_base.device).view(1, T, 1)

        z = None
        for gated_x_attn in self.gatedXAttn_layers:
            z = gated_x_attn(z_base, z_supp, attn_mask=allow, q_mask=base_mask, kv_mask=z_mask)

        logits = self.projector(z)
        return (logits, kd_outputs) if use_kd else logits

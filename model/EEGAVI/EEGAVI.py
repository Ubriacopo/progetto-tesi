from typing import Optional

import torch
from einops import rearrange
from torch import nn

from model.layer.attention.x_attention import GatedCrossAttentionBlock
from model.layer.base import ModalContextEncoder
from model.layer.modality_stream import ModalityStream
from model.EEGAVI.transforms import media_locs_single_item

import lightning as L


def pool_mod_to_text_bins(X_mod, M_mod, T_txt):
    """
    X_mod: (B, T_mod, D)        # one embedding per modality step
    M_mod: (B, T_mod) bool      # True=valid step
    returns:
      Z: (B, T_txt, D)          # pooled per text bin
      M: (B, T_txt) bool        # True if bin has any valid steps
    """
    B, T_mod, P, D = X_mod.shape
    device = X_mod.device
    # proportional mapping i -> bin
    i = torch.arange(T_mod, device=device)  # (T_mod,)
    bin_idx = torch.floor(i * (T_txt / T_mod)).clamp(max=T_txt - 1).long()  # (T_mod,)

    # build (B, T_txt) counts and (B, T_txt, D) sums using masks
    Z = X_mod.new_zeros(B, T_txt, P, D)
    C = X_mod.new_zeros(B, T_txt, 1, 1)  # counts per bin

    # loop over bins (T_txt is tiny = 8, so this is fine)
    for t in range(T_txt):
        sel_t = (bin_idx == t)  # (T_mod,)
        if not sel_t.any():
            continue

        # X_mod: (B, T_mod, F, D), M_mod: (B, T_mod) bool
        # inside the loop:
        X_t = X_mod[:, sel_t, :, :]  # (B, n_t, F, D)
        M_t = M_mod[:, sel_t].view(B, -1, 1, 1)  # (B,n_t,1,1)
        X_t = X_t * M_t
        Z[:, t, :, :] = X_t.sum(dim=1)  # (B, F, D)
        C[:, t, :, :] = M_t.sum(dim=1, dtype=Z.dtype)  # (B, F, 1)
        # ... then Z = Z / C, Mask per bin = (C.squeeze(-1).sum(dim=-1) > 0)
    # compute Mb from *raw* counts (no clamp!)
    Mb = (C.squeeze(-1).squeeze(-1) > 0)
    # safe division: clamp only where Mb is True
    denom = C.clone()
    denom[Mb.unsqueeze(-1).unsqueeze(-1)] = torch.clamp_min(denom[Mb.unsqueeze(-1).unsqueeze(-1)], 1.0)
    Z = torch.where(Mb.unsqueeze(-1).unsqueeze(-1), Z / denom, Z * 0)

    return Z, Mb


def interval_overlap_weights(T_src: int, T_tgt: int, device=None):
    """
    Returns W of shape (T_tgt, T_src) with non-negative rows summing to 1
    that describe how much of each source interval contributes to each target interval.
    Intervals are uniform partitions of [0, 1].
    """
    device = device or "cpu"
    # Interval edges
    src_edges = torch.linspace(0, 1, T_src + 1, device=device)
    tgt_edges = torch.linspace(0, 1, T_tgt + 1, device=device)

    # Interval spans
    src_st, src_en = src_edges[:-1], src_edges[1:]
    tgt_st, tgt_en = tgt_edges[:-1], tgt_edges[1:]

    # Compute pairwise overlaps
    # overlap(t,s) = max(0, min(t_en, s_en) - max(t_st, s_st))
    t_st = tgt_st[:, None]  # (T_tgt, 1)
    t_en = tgt_en[:, None]  # (T_tgt, 1)
    s_st = src_st[None, :]  # (1, T_src)
    s_en = src_en[None, :]  # (1, T_src)

    overlap = (torch.minimum(t_en, s_en) - torch.maximum(t_st, s_st)).clamp(min=0)  # (T_tgt, T_src)

    # Normalize rows to sum to 1 where possible
    row_sum = overlap.sum(dim=1, keepdim=True)
    W = torch.where(row_sum > 0, overlap / row_sum, overlap)  # rows with all-zero stay zero
    return W  # (T_tgt, T_src)


def remap_with_overlap(X_src, M_src, T_tgt):
    """
    X_src: (B, T_src, P, D) or (B, T_src, D)  # time-major features
    M_src: (B, T_src) bool                    # valid steps
    T_tgt: int
    Returns:
      X_tgt: (B, T_tgt, P, D) or (B, T_tgt, D)
      M_tgt: (B, T_tgt) bool
    """
    B = X_src.size(0);
    T_src = X_src.size(1)
    keep_P = (X_src.dim() == 4)
    if not keep_P:  # upgrade to a (P) axis for unified math
        X_src = X_src.unsqueeze(2)  # (B, T_src, 1, D)

    P, D = X_src.size(2), X_src.size(3)
    W = interval_overlap_weights(T_src, T_tgt, device=X_src.device)  # (T_tgt, T_src)

    # Mask invalid source steps
    Wb = W[None, :, :] * M_src[:, None, :].to(X_src.dtype)  # (B, T_tgt, T_src)

    # Denominator per target bin (how much valid mass contributed)
    denom = Wb.sum(dim=2, keepdim=True).clamp(min=1e-9)  # (B, T_tgt, 1)

    # Weighted sum over source time
    # Expand Wb to broadcast over P,D
    Wb_exp = Wb[:, :, :, None, None]  # (B, T_tgt, T_src, 1, 1)
    X_exp = X_src[:, None, :, :, :]  # (B, 1, T_src, P, D)
    denom = denom.squeeze(-1)
    X_tgt = (Wb_exp * X_exp).sum(dim=2) / denom[:, :, None, None]  # (B, T_tgt, P, D)

    # Target mask: bin valid if it received any valid mass
    M_tgt = (Wb.sum(dim=2) > 0)  # (B, T_tgt)

    if not keep_P:
        X_tgt = X_tgt.squeeze(2)  # (B, T_tgt, D)
    return X_tgt, M_tgt


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

        base = x[self.pivot_modality.get_code()]
        base_mask = None

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

        supports = []
        masks = []
        for z, mask in zip(z_supports, z_masks):
            Z_vid, M_vid = remap_with_overlap(z, mask, 32)
            supports.append(Z_vid)
            masks.append(M_vid)
        # TODO: Get media locs for z base
        media_locations = media_locs_single_item(z_base.shape[0], 1, z_base.device)
        z_supp: torch.Tensor = torch.cat(z_supports, dim=1)

        z = None
        for gated_x_attn in self.gatedXAttn_layers:
            # todo passa mod masks
            z = gated_x_attn(z_base, z_supp, media_locations=media_locations)

        logits = self.projector(z)
        return (logits, kd_outputs) if use_kd else logits

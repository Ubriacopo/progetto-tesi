from typing import Optional

import torch
from einops import rearrange
from torch import nn

from model.layer.attention.x_attention import GatedCrossAttentionBlock
from model.layer.base import ModalContextEncoder
from model.layer.modality_stream import ModalityStream
from model.EEGAVI.transforms import media_locs_single_item

import lightning as L


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

        base = x[self.pivot_modality.get_code()]
        base_mask = None

        z_base = self.pivot_modality(base, mask=base_mask, use_kd=use_kd)
        if isinstance(z_base, tuple):
            kd_outputs[self.pivot_modality.get_code()] = z_base[1]
            z_base = z_base[0]

        for adapter in self.supporting_modalities:
            key = adapter.get_code()
            supp = x[key]
            mod_mask = None  # Masks are already pre-computed in structure dict?

            z_supp = adapter(supp, mask=mod_mask, use_kd=use_kd)
            if isinstance(z_supp, tuple):
                # Store the KD output to return later
                kd_outputs[key] = z_supp[1]
                # Now we can really get the embeddings
                z_supp = z_supp[0]

            if self.modality_encoder is not None:
                z_supp = self.modality_encoder(z_supp, modality=key)

            z_supports.append(z_supp)

        # TODO: Get media locs for z base
        media_locations = media_locs_single_item(z_base.shape[0], 1, z_base.device)
        z_supp: torch.Tensor = torch.cat(z_supports, dim=1)
        if len(z_supp.shape) == 3:
            # (b, T*F, D) (Case of no time series used).
            z_supp = rearrange(z_supp, "b (T F) d -> b T F d", T=1)

        z = None
        for gated_x_attn in self.gatedXAttn_layers:
            z = gated_x_attn(z_base, z_supp, media_locations=media_locations)

        logits = self.projector(z)
        return (logits, kd_outputs) if use_kd else logits

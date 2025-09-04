import torch
from einops import rearrange
from torch import nn

from common.model.embedding.embedder_adapter import EmbedderAdapter
from common.model.layers.attention.x_attention import GatedCrossAttentionBlock
from models.EEGAVI.transforms import media_locs_single_item


class EEGAVI(nn.Module):
    def __init__(self,
                 target_size: int,
                 cross_attention_blocks: int,

                 final_projector: nn.Module,

                 pivot_modality: tuple[str, EmbedderAdapter],  # EEG
                 supporting_modalities: list[tuple[str, EmbedderAdapter]],

                 ):
        super(EEGAVI, self).__init__()

        self.pivot_modality = pivot_modality
        self.supporting_modalities = supporting_modalities

        self.gatedXAttn_layers = nn.ModuleList([
            GatedCrossAttentionBlock(dim=target_size, dim_latent=target_size)
            for _ in range(cross_attention_blocks)
        ])
        self.projector = final_projector

    def forward(self, x: dict):
        modalities_idx = list(zip(*x["order"]))[0]
        use_kd = "kd" in x and x["kd"]
        # TODO Vedere poi
        mask = x["mask"] if "mask" in x else None

        # Base Modality first
        kd_outputs = {}
        z_supports = []

        base = x[self.pivot_modality[0]]
        z_base = self.pivot_modality[1](base, use_kd)
        if isinstance(z_base, tuple):
            kd_outputs[self.pivot_modality[0]] = z_base[1]
            z_base = z_base[0]

        for key, adapter in self.supporting_modalities:
            supp = x[key]
            z_supp = adapter(supp, use_kd)
            if isinstance(z_supp, tuple):
                kd_outputs[key] = z_supp[1]
                z_supp = z_supp[0]
            z_supports.append(z_supp)

        # TODO: Get media locs for z base
        media_locations = media_locs_single_item(z_base[0], 1, z_base.device)

        z_supp = torch.cat(z_supports, dim=0)
        if len(z_supp.shape) == 3:
            # (b, T*F, D) (Case of no time series used).
            z_supp = rearrange(z_supp, "b (T F) d -> b T F d", T=1)

        z = None
        for gated_x_attn in self.gatedXAttn_layers:
            z = gated_x_attn(z_base, z_supp, media_locations=media_locations)

        logits = self.projector(z)
        return logits,

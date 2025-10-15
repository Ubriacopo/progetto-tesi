import dataclasses
from dataclasses import asdict
from typing import Optional

import torch
from torch import nn

from main.model.EEGAVI.utils import remap_with_overlap
from main.model.layer.attention.x_attention import GatedXAttentionCustomArgs, GatedXAttentionBlock
from main.model.layer.base import ModalContextEncoder
from main.model.layer.modality_stream import ModalityStream
from main.utils.data import MaskedValue, KdMaskedValue


@dataclasses.dataclass
class EegBaseModelOutputs:
    embeddings: torch.Tensor
    kd_outs: dict[str, MaskedValue]
    multimodal_outs: dict[str, MaskedValue]


class FusionPooling(nn.Module):
    # noinspection PyMethodMayBeStatic
    def forward(self, z: torch.Tensor, mask=None) -> torch.Tensor:
        norm_factor = mask.float().sum(dim=-1, keepdim=True)
        norm_factor = norm_factor.clamp_min(1e-6)
        z = (z * mask.unsqueeze(-1)).sum(dim=-2) / norm_factor
        return z


class EegBaseModel(nn.Module):
    def __init__(self,
                 output_size: int,
                 pivot: ModalityStream, supports: list[ModalityStream],
                 xattn_blocks: int | list[GatedXAttentionCustomArgs],
                 remap_timesteps: int,
                 drop_p: float = 0.0, use_modality_encoder: bool = True):
        super().__init__()

        self.pivot = pivot
        self.supports = supports

        self.modality_encoder: Optional[ModalContextEncoder] = None
        if use_modality_encoder:
            modality_mappings = {e.get_code(): i for i, e in enumerate(supports)}
            self.modality_encoder = ModalContextEncoder(supports[0].output_size, modality_mappings)

        self.gatedXAttn_layers = nn.ModuleList(self.build_xattn_blocks(xattn_blocks))

        self.remap_timesteps: int = remap_timesteps
        self.fusion_pooling: nn.Module = self.build_fusion_pooling()
        self.fused_norm = nn.LayerNorm(output_size)
        self.drop_p = drop_p

    # noinspection PyMethodMayBeStatic
    def build_fusion_pooling(self):
        return FusionPooling()

    def process_pivot(self, x: MaskedValue, use_kd: bool) -> MaskedValue | KdMaskedValue:
        return self.pivot_modality(x["data"], mask=x.get("mask", None), use_kd=use_kd)

    def process_modality(self, x: MaskedValue, idx: torch.Tensor, b: int, modality: ModalityStream, use_kd: bool):
        output = {}

        data = x["data"]
        mask = x.get("mask", None)
        pass_mask = mask[idx] if mask is not None else None

        y: MaskedValue | KdMaskedValue = modality(data[idx], mask=pass_mask, use_kd=use_kd)

        if "kd" in y:
            y: KdMaskedValue
            kd_data = y.pop("kd")
            kd = torch.zeros(y["data"].shape[0], *kd_data["data"].shape[1:], device=kd_data["data"].device)
            kd[idx] = kd_data["data"]

            kd_mask = torch.zeros(y["data"].shape[0], *kd_data["data"].shape[1:], device=kd_data["data"].device)
            kd_mask[idx] = kd_data["mask"]

            output["kd"] = MaskedValue(data=kd, mask=kd_mask)

        y: MaskedValue
        z = y["data"]
        if self.modality_encoder is not None:
            z = self.modality_encoder(z, modality=modality.get_code())

        # TODO Revisiona questa parte che forse non funziona correttamente:
        padded_z = z.new_zeros(b, *z.shape[1:])
        padded_z[idx] = z[idx]

        padded_mask = None
        if mask is None:
            # Even if masking is disabled for new created rows the mask has to be generated else it would break everyhting.
            padded_mask = torch.zeros(b, padded_z.size(1), dtype=torch.bool, device=padded_z.device)
            padded_mask[idx] = True
        else:
            padded_mask = mask.new_zeros(b, padded_z.size(1), dtype=torch.bool, device=padded_z.device)
            padded_mask[idx] = mask[idx]

        z, mask = remap_with_overlap(padded_z, padded_mask, self.remap_timesteps)
        z = mask * z[:, :, None, None]
        mask = mask[:, :, None].expand(-1, -1, z.shape[2])

        output["data"] = z
        output["mask"] = mask

        return output

    def forward(self, x: dict, use_kd: bool = False, return_dict: bool = False) -> EegBaseModelOutputs | dict:
        kd_outs: dict = {}
        multimodal_outs: dict = {}
        device = x[self.pivot.get_code()]["data"].device

        pivot_out = self.process_pivot(x[self.pivot.get_code()], use_kd=use_kd)
        if "kd" in pivot_out:  # KD is enabled
            kd_outs[self.pivot.get_code()] = pivot_out.pop("kd")
        multimodal_outs[self.pivot.get_code()] = pivot_out

        B = x[self.pivot.get_code()]["data"].shape[0]  # Batch size
        keep = self.select_keeps(B, device)

        supports = []
        masks = []

        for modality_idx, modality in enumerate(self.supports):
            code = modality.get_code()
            idx = keep[:, modality_idx].nonzero(as_tuple=True)[0]
            modality_out = self.process_modality(x[code], idx=idx, modality=modality, use_kd=use_kd)

            multimodal_outs[code] = modality_out

        support = torch.cat(supports, dim=2)
        masks = torch.cat(masks, dim=2)

        allow = self.remask(supp=support, device=support.device)
        z: torch.Tensor = pivot_out["data"]
        for gated_x_attn in self.gatedXAttn_layers:
            z = gated_x_attn(z, support, attn_mask=allow, q_mask=pivot_out, kv_mask=masks)

        z = self.fusion_pooling(z, mask=pivot_out["mask"])
        z = self.fused_norm(z)

        return_object = EegBaseModelOutputs(embeddings=z, kd_outs=kd_outs, multimodal_outs=multimodal_outs)
        return return_object if not return_dict else asdict(return_object)

    def select_keeps(self, b: int, device):
        # todo pass vals and see if correct
        number_modalities = len(self.supports)
        if (not self.training) or self.drop_p <= 0:
            return torch.ones(b, number_modalities, device=device)

        keep = torch.bernoulli(torch.full((b, number_modalities), 1 - self.drop_p, device=device)).bool()
        dead = ~keep.any(1)
        if dead.any():
            # We force at least one modality to always be on.
            keep[dead, torch.randint(0, number_modalities, (dead.sum(),), device=device)] = True

        return keep

    def build_xattn_blocks(self, xattn_blocks: int | list[GatedXAttentionCustomArgs]) -> list[nn.Module]:
        """
        For how GatedXAttention works the dim and dim_latent are fixed (they do not change).
        :param xattn_blocks:
        :return:
        """
        modules: list[nn.Module] = []
        if isinstance(xattn_blocks, list):
            for config in xattn_blocks:
                xattn_block = GatedXAttentionBlock(self.pivot_latent_size, self.supporting_latent_size, *asdict(config))
                modules.append(xattn_block)

        elif isinstance(xattn_blocks, int):
            for _ in range(xattn_blocks):
                xattn_block = GatedXAttentionBlock(self.pivot_latent_size, self.supporting_latent_size)
                modules.append(xattn_block)

        return modules

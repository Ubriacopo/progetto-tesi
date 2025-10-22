import dataclasses
import logging
from dataclasses import asdict
from typing import Optional

import torch
from einops import repeat
from torch import nn

from main.model.EEGAVI.utils import remap_with_overlap
from main.model.neegavi.blocks import ModalityStream, ModalContextEncoder
from main.model.neegavi.pooling import MaskedPooling
from main.model.neegavi.xattention import GatedXAttentionCustomArgs, GatedXAttentionBlock
from main.utils.data import MaskedValue, KdMaskedValue


@dataclasses.dataclass
class EegBaseModelOutputs:
    embeddings: torch.Tensor
    kd_outs: dict[str, MaskedValue]
    multimodal_outs: dict[str, MaskedValue]


@dataclasses.dataclass
class WeaklySupervisedEegBaseModelOutputs(EegBaseModelOutputs):
    pred: torch.Tensor


class EegBaseModel(nn.Module):
    def __init__(self,
                 output_size: int,
                 pivot: ModalityStream, supports: list[ModalityStream],
                 xattn_blocks: int | list[GatedXAttentionCustomArgs],
                 remap_timesteps: int,
                 drop_p: float = 0.0, use_modality_encoder: bool = True):
        super().__init__()

        self.output_size = output_size
        if len(supports) == 0:
            raise ValueError("For EegBaseModel, supports must not be empty")

        self.latent_output_size = supports[0].output_size
        for i in supports:
            if i.output_size != self.latent_output_size:
                error_msg = (
                    f"Output size of support {i.code} ({i.output_size}) does not match extracted size "
                    f"of {supports[0].code} ({self.latent_output_size})"
                )

                logging.error(error_msg)
                raise ValueError(error_msg)

        self.pivot = pivot
        self.supports: nn.ModuleList[ModalityStream] = nn.ModuleList(supports)

        self.modality_encoder: Optional[ModalContextEncoder] = None
        if use_modality_encoder:
            modality_mappings = {e.get_code(): i for i, e in enumerate(supports)}
            self.modality_encoder = ModalContextEncoder(supports[0].output_size, modality_mappings)

        self.gatedXAttn_layers = nn.ModuleList(self.build_xattn_blocks(xattn_blocks))

        self.remap_timesteps: int = remap_timesteps
        self.fusion_pooling: nn.Module = self.build_fusion_pooling()
        self.fused_norm = nn.LayerNorm(output_size)
        self.drop_p = drop_p

    @staticmethod
    def remask(supp: torch.Tensor, device):
        b, T, F, D = supp.shape
        key_time_idx = torch.arange(T, device=device).repeat_interleave(F)
        key_time_idx = repeat(key_time_idx, "D -> a b D", a=1, b=1)
        # allow[q_t, k] = (time(k) <= q_t)
        allow = key_time_idx.view(1, 1, -1) <= torch.arange(T, device=device).view(1, T, 1)
        return allow

    # noinspection PyMethodMayBeStatic
    def build_fusion_pooling(self):
        return MaskedPooling()

    def process_pivot(self, x: MaskedValue, use_kd: bool) -> MaskedValue | KdMaskedValue:
        return self.pivot(x["data"], mask=x.get("mask", None), use_kd=use_kd)

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

            kd_mask = torch.zeros(y["data"].shape[0], *kd_data["data"].shape[1:-1], device=kd_data["data"].device)
            kd_mask = kd_mask.bool()
            kd_mask[idx] = kd_data["mask"]

            output["kd"] = MaskedValue(data=kd, mask=kd_mask)

        y: MaskedValue
        z = y["data"]
        if self.modality_encoder is not None:
            z = self.modality_encoder(z, modality=modality.get_code())

        # This section remaps the input to its original batch size. This is not needed if idx has range == b.
        # If we don't add padding the model breaks (suppose for drop we take only 6 of 24, that would not rebatch correctly)
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
        mask = repeat(mask, "b t -> b t P", P=z.shape[2])

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

        b = pivot_out["data"].shape[0]

        for modality_idx, modality in enumerate(self.supports):
            modality: ModalityStream
            code = modality.get_code()
            idx = keep[:, modality_idx].nonzero(as_tuple=True)[0]
            modality_out = self.process_modality(x[code], idx=idx, modality=modality, use_kd=use_kd, b=b)

            if "kd" in modality_out:
                kd_outs[code] = modality_out.pop("kd")
            multimodal_outs[code]: KdMaskedValue = modality_out
            # For faster indexing to make cat
            supports.append(modality_out["data"])
            masks.append(modality_out["mask"])

        support = torch.cat(supports, dim=2)
        masks = torch.cat(masks, dim=2)

        allow = self.remask(supp=support, device=support.device)
        z: torch.Tensor = pivot_out["data"]
        for gated_x_attn in self.gatedXAttn_layers:
            z = gated_x_attn(z, support, attn_mask=allow, q_mask=pivot_out["mask"], kv_mask=masks)

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
                xattn_block = GatedXAttentionBlock(self.output_size, self.latent_output_size, *asdict(config))
                modules.append(xattn_block)

        elif isinstance(xattn_blocks, int):
            for _ in range(xattn_blocks):
                xattn_block = GatedXAttentionBlock(self.output_size, self.latent_output_size)
                modules.append(xattn_block)

        return modules


class WeaklySupervisedEegBaseModel(nn.Module):
    def __init__(self, eeg_base_model: EegBaseModel, hidden_size: int, output_size: int):
        super(WeaklySupervisedEegBaseModel, self).__init__()
        self.base_model = eeg_base_model
        self.prediction_head = nn.Sequential(
            nn.Linear(eeg_base_model.output_size, hidden_size),
            nn.ReLU(),
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, x: dict, use_kd: bool = False, return_dict: bool = False):
        outs: EegBaseModelOutputs = self.base_model(x, use_kd=use_kd, return_dict=False)
        pred = self.prediction_head(outs.embeddings)
        o = WeaklySupervisedEegBaseModelOutputs(pred=pred, **vars(outs))
        return o if not return_dict else asdict(o)

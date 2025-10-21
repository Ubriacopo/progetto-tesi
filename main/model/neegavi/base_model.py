import logging
from dataclasses import asdict
from typing import Optional, Literal

import torch
from einops import repeat, rearrange
from torch import nn

from main.model.EEGAVI.base_model import EegBaseModelOutputs, WeaklySupervisedEegBaseModelOutputs
from main.model.layer.base import ModalContextEncoder
from main.model.neegavi.blocks import MaskedPooling, ModalityStream
from main.model.neegavi.xattention import GatedXAttentionCustomArgs, GatedXAttentionBlock
from main.utils.data import MaskedValue, KdMaskedValue


class Model(nn.Module):
    pass


class EegInterAviModel(nn.Module):
    def __init__(self, output_size: int,
                 pivot: ModalityStream, supports: list[ModalityStream],
                 xattn_blocks: int | list[GatedXAttentionCustomArgs],
                 drop_p: float = 0.0, use_modality_encoder: bool = True,
                 fusion_pooling: nn.Module = MaskedPooling()):
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

        self.drop_p = drop_p
        self.allow_modality: Literal['window', 'causal'] = 'window'
        self.window_seconds = 2

        self.fusion_pooling: nn.Module = fusion_pooling

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

    def select_keep_modality_rows(self, batch_size: int, device):
        num_modalities = len(self.supports)
        if (not self.training) or self.drop_p <= 0:
            return torch.ones(batch_size, num_modalities, device=device)
        # TODO: Da fare in futuro (robustezza contro mod mancante)
        # keep = torch.bernoulli(torch.full((b, number_modalities), 1 - self.drop_p, device=device)).bool()
        return torch.ones(batch_size, num_modalities, device=device)

    def process_modality(self, x: MaskedValue, idx: torch.Tensor, b: int, modality: ModalityStream, use_kd: bool):
        output = {}
        data = x["data"]
        t = data.shape[1]  # (b, t, ...) Time is always second axis

        mask = x.get("mask", None)
        # For the moment ignore idx
        y: MaskedValue | KdMaskedValue = modality(data, mask=mask, use_kd=use_kd)

        if "kd" in y:
            output["kd"] = y.pop("kd")
            # TODO: Poi dovrai ricostruire shape corrette ma ora stiamo senza idx

        y: MaskedValue
        z = y["data"]
        if self.modality_encoder is not None:
            z = self.modality_encoder(z, modality=modality.get_code())

        # Time mask
        t_mod = torch.arange(t, device=data.device) * modality.time_step_length
        # Flatten [b, T, M, D] thus update mask and t_mod as-well
        m = z.shape[2]

        z = rearrange(z, "b t m d -> b (t m) d")
        t_mod = repeat(t_mod, "t -> b (t m)", b=b, m=m)
        mask = repeat(mask, "b t -> b (t m)", m=m)
        return output | {"data": z, "mask": mask, "t_mod": t_mod}

    def build_allow_mask(self, t_q: torch.Tensor, t_kv: torch.Tensor):
        tq = t_q.unsqueeze(-1)  # [B, Tq, 1]
        tk = t_kv.unsqueeze(1)  # [B, 1, Tk]

        if self.allow_modality == "window":
            return (tk - tq).abs() <= self.window_seconds
        if self.allow_modality == "causal":
            return tk <= tq
        raise ValueError(f"Unknown mode: {self.allow_modality}")

    def forward(self, x: dict, use_kd: bool = False, return_dict: bool = False):
        kd_outputs, multimodal_outs = {}, {}

        pivot_x = x[self.pivot.get_code()]
        device = pivot_x["data"].device

        b = pivot_x["data"].shape[0]  # Batch size
        t_pivot = torch.arange(pivot_x["data"].shape[1], device=device) * self.pivot.time_step_length
        t_pivot = repeat(t_pivot, "t -> b t", b=b)

        pivot_output = self.pivot(pivot_x["data"], mask=pivot_x.get("mask", None), use_kd=use_kd)

        if "kd" in pivot_output:
            kd_outputs[self.pivot.get_code()] = pivot_output.pop("kd")
        multimodal_outs[self.pivot.get_code()] = pivot_output

        keep = self.select_keep_modality_rows(b, device)
        supports, masks, t_mods = [], [], []

        modality: ModalityStream
        for idx, modality in enumerate(self.supports):
            code = modality.get_code()
            filtered_idx = keep[:, idx].nonzero(as_tuple=True)[0]
            modality_out = self.process_modality(x[code], idx=filtered_idx, modality=modality, use_kd=use_kd, b=b)

            if "kd" in modality_out:
                kd_outputs[code] = modality_out.pop("kd")
            multimodal_outs[code]: KdMaskedValue = modality_out

            supports.append(modality_out["data"])
            masks.append(modality_out["mask"])
            t_mods.append(modality_out["t_mod"])

        support = torch.cat(supports, dim=1)
        masks = torch.cat(masks, dim=1)
        t_mod = torch.cat(t_mods, dim=1)

        allow = self.build_allow_mask(t_pivot, t_mod)
        z: torch.Tensor = pivot_output["data"]
        for gated_x_attn in self.gatedXAttn_layers:
            z = gated_x_attn(z, support, attn_mask=allow, q_mask=pivot_output["mask"], kv_mask=masks)

        if self.fusion_pooling is not None:
            z = self.fusion_pooling(z, mask=pivot_output["mask"])

        return_object = EegBaseModelOutputs(embeddings=z, kd_outs=kd_outputs, multimodal_outs=multimodal_outs)
        return return_object if not return_dict else asdict(return_object)


class WeaklySupervisedNEEEGBaseModel(nn.Module):
    def __init__(self, eeg_base_model: EegInterAviModel, hidden_size: int, output_size: int):
        super(WeaklySupervisedNEEEGBaseModel, self).__init__()
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

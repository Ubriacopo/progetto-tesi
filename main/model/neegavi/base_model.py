import logging
from dataclasses import asdict
from typing import Optional, Literal

import torch
from einops import repeat, rearrange
from torch import nn

from main.model.neegavi.blocks import ModalityStream, ModalContextEncoder
from main.model.neegavi.pooling import MaskedPooling
from main.model.neegavi.utils import EegBaseModelOutputs, WeaklySupervisedEegBaseModelOutputs
from main.model.neegavi.xattention import GatedXAttentionCustomArgs, GatedXAttentionBlock
from main.utils.data import MaskedValue, KdMaskedValue


class EegInterAviModel(nn.Module):
    def __init__(self, output_size: int,
                 pivot: ModalityStream, supports: list[ModalityStream],
                 xattn_blocks: int | list[GatedXAttentionCustomArgs],
                 drop_p: float = 0.0, use_modality_encoder: bool = True,
                 fusion_pooling: nn.Module = MaskedPooling()):
        """
        :param pivot: Main modality pipeline that acts as receptor is xattn (q), It is always required
        :param supports: Other modalities and their pipeline. Once set for a model you can't of course change them.
                         During inference and training they all are optional.
        :param xattn_blocks: Number of xattn blocks to apply after each modality has been processed.
                             Can also be a list of different configurations for the GatedXAttention blocks.
        :param drop_p: Drop probability for each modality for each sample in batch

        :param use_modality_encoder: Whether to use modality encoder or not. IF flagged to true a weight vector is
                                     added to each supporting modality before making concat of the embeddings for xattn.
        :param fusion_pooling: Pooling logic to get to output shape. Optional. By default, is norm over mask.
        """
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

        self.pivot: ModalityStream = pivot
        self.supports: nn.ModuleList[ModalityStream] = nn.ModuleList(supports)
        self.modality_encoder: Optional[ModalContextEncoder] = None
        if use_modality_encoder:
            modality_mappings = {e.get_code(): i for i, e in enumerate(supports)}
            self.modality_encoder = ModalContextEncoder(supports[0].output_size, modality_mappings)

        self.gatedXAttn_layers = nn.ModuleList(self.build_xattn_blocks(xattn_blocks))

        self.drop_p = drop_p
        self.allow_modality: Literal['window', 'causal'] = 'window'
        # TODO Window units (each block has same unit of measurement)
        self.past_window_units = 2

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

        # todo verifica
        keep = torch.bernoulli(torch.full((batch_size, num_modalities), 1 - self.drop_p, device=device)).bool()
        dead = ~keep.any(1)

        if dead.any():
            # We force at least one modality to always be on.
            keep[dead, torch.randint(0, num_modalities, (dead.sum(),), device=device)] = True

        return keep

    # TODO si semplifica -> ogni mod uniformata a stesso timestep di EEG
    def process_modality(self, x: MaskedValue, idx: torch.Tensor, b: int, modality: ModalityStream, use_kd: bool):
        output = {}

        device = x["data"].device
        # (b, t, ...) Time is always second axis
        t = x["data"].shape[1]

        mask = x.get("mask", None)
        # For the moment ignore idx
        y: MaskedValue | KdMaskedValue = modality(x["data"], mask=x.get("mask", None), use_kd=use_kd)
        if "kd" in y:
            kd_data = y.pop("kd")

            kd = torch.zeros(b, *kd_data["data"].shape[1:], device=kd_data["data"].device)
            kd[idx] = kd_data["data"]
            kd_mask = torch.zeros(b, *kd_data["mask"].shape[1:], device=kd_data["data"].device).bool()
            kd_mask[idx] = kd_data["mask"]

            output["kd"] = MaskedValue(data=kd, mask=kd_mask)

        y: MaskedValue  # We removed the kd part from y
        z = y["data"]
        if self.modality_encoder is not None:
            z = self.modality_encoder(z, modality=modality.get_code())

        # Time mask
        t_mod = torch.arange(t, device=device)
        # Flatten [b, T, M, D] thus update mask and t_mod as-well
        m = z.shape[2]

        z = rearrange(z, "b t m d -> b (t m) d")
        t_mod = repeat(t_mod, "t -> b (t m)", b=b, m=m)
        mask = repeat(mask, "b t -> b (t m)", m=m)

        full_z = z.new_zeros(b, *z.shape[1:])
        # Restore original indexes elements
        full_z[idx] = z[idx]

        full_mask = torch.zeros(b, full_z.size(1), dtype=torch.bool, device=full_z.device)
        full_mask[idx] = True if mask is None else mask[idx]
        return output | {"data": full_z, "mask": full_mask, "t_mod": t_mod}

    def build_allow_mask(self, t_q: torch.Tensor, t_kv: torch.Tensor):
        tq = t_q.unsqueeze(-1)  # [B, Tq, 1]
        tk = t_kv.unsqueeze(1)  # [B, 1, Tk]

        if self.allow_modality == "window":
            return (tk - tq).abs() <= self.past_window_units
        if self.allow_modality == "causal":
            return tk <= tq
        raise ValueError(f"Unknown mode: {self.allow_modality}")

    def forward(self, x: dict, use_kd: bool = False, return_dict: bool = False):
        """
        :param x:
        :param use_kd:
        :param return_dict:
        :return:
        """

        kd_outputs, multimodal_outs = {}, {}

        pivot_x = x[self.pivot.get_code()]
        device = pivot_x["data"].device

        b = pivot_x["data"].shape[0]  # Batch size
        t_pivot = torch.arange(pivot_x["data"].shape[1], device=device)
        t_pivot = repeat(t_pivot, "t -> b t", b=b)

        pivot_output = self.pivot(pivot_x["data"], mask=pivot_x.get("mask", None), use_kd=use_kd)

        if "kd" in pivot_output:
            kd_outputs[self.pivot.get_code()] = pivot_output.pop("kd")

        multimodal_outs[self.pivot.get_code()] = pivot_output
        # Pre-compute indices of what modalities to keep and drop
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

        out_size: int = self.latent_output_size
        # In case no modality passes through we have to still create an empty vector
        support = torch.cat(supports, dim=1) if len(supports) != 0 else torch.zeros(b, 1, out_size, device=device)
        masks = torch.cat(masks, dim=1) if len(masks) != 0 else torch.zeros(b, 1, device=device)
        t_mod = torch.cat(t_mods, dim=1) if len(t_mods) != 0 else torch.zeros(b, 1, device=device)

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

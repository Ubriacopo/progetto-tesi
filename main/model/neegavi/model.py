from abc import abstractmethod, ABC
from dataclasses import asdict
from typing import Optional

import torch
from einops import repeat
from torch import nn

from main.model.neegavi.blocks import ModalityStream
from main.model.neegavi.utils import EegBaseModelOutputs
from main.utils.data import MaskedValue
from main.utils.logging import make_logger


class EegInterAviModel(nn.Module, ABC):
    KD_KEY = "kd"

    def __init__(self, pivot: ModalityStream, supports: list[ModalityStream], drop_p: float = .0):
        super(EegInterAviModel, self).__init__()
        self.logger = make_logger(self.__class__.__name__)

        self.pivot: ModalityStream = pivot
        self.supports: list[ModalityStream] = supports
        self.supports_feature_size: Optional[int] = None
        self.check_supports()

        self.drop_p: float = drop_p

    def check_supports(self):
        if len(self.supports) == 0:
            error_message = "Supports cannot be empty. At least one support must be provided."
            self.logger.error(error_message)
            raise ValueError(error_message)

        self.supports_feature_size: int = self.supports[0].output_size
        for support in self.supports:
            if support.output_size != self.supports_feature_size:
                error_msg = (f"Output size of support {support.code} ({support.output_size}) does not "
                             f"match extracted size of {self.supports[0].code} ({self.latent_output_size})")
                self.logger.error(error_msg)
                raise ValueError(error_msg)

    def rand_select_keep_modality_rows(self, batch_size: int, device, ensure_one: bool = False):
        if (not self.training) or self.drop_p <= 0:
            return torch.ones(batch_size, len(self.supports), device=device)
        keep = torch.bernoulli(torch.full((batch_size, len(self.supports)), 1 - self.drop_p, device=device)).bool()

        dead = ~keep.any(1)
        # TODO decidi se fare ensure di one
        if ensure_one and dead:
            summed_dead = dead.sum().item()
            keep[dead, torch.randint(0, len(self.supports), (summed_dead,), device=device)] = True

        return keep

    def process_pivot(self):
        pass

    @abstractmethod
    def align_pivot_time_to_support_time(self):
        pass

    def process_modality(self, x: MaskedValue, idx: torch.Tensor, b: int, modality: ModalityStream, use_kd: bool):
        output = dict()
        modality_data, modality_mask = x["data"], x.get("mask", None)
        _, t = modality_data.shape[0:2]

        if modality_mask is not None:
            modality_mask = modality_mask.bool()


    def forward(self, x: dict, use_kd: bool = False, return_dict: bool = False):
        # Where outputs are partitioned for later use.
        out = EegBaseModelOutputs(torch.empty(), {}, {})

        # Pivot modality elaboration

        pivot_x = x[self.pivot.get_code()]
        pivot_data, pivot_mask = pivot_x["data"], pivot_x.get("mask", None)
        # Device to always use the same
        device = pivot_data.device
        b, t = pivot_data.shape[0:2]

        # Time of pivot
        time_pivot = torch.arange(t, device=device)
        time_pivot = repeat(time_pivot, "t -> b t", b=b)

        pivot_out = self.pivot(pivot_data, mask=pivot_mask, use_kd=use_kd)
        if self.KD_KEY in pivot_out:
            out.kd_outs[self.pivot.get_code()] = pivot_out.pop(self.KD_KEY)
        out.multimodal_outs[self.pivot.get_code()] = pivot_out

        # Supporting modalities elaboration
        keep = self.rand_select_keep_modality_rows(b, device)
        supports, masks, t_mods = [], [], []
        for idx, modality in enumerate(self.supports):
            modality_code = modality.get_code()
            # TODO why [0]?
            filtered_idx = keep[:, idx].nonzero(as_tuple=True)[0]
            modality_out = self.process_modality(
                x[modality_code], idx=filtered_idx, modality=modality, use_kd=use_kd, b=b
            )

            if self.KD_KEY in modality_out:
                out.kd_outs[modality_code] = modality_out.pop(self.KD_KEY)
            out.multimodal_outs[modality_out] = modality_out

            supports.append(modality_out["data"])
            masks.append(modality_out["mask"])
            t_mods.append(modality_out["t_mod"])

            out_size: int = self.supports_feature_size
            # In case no modality passes through we have to still create an empty vector
            support = torch.cat(supports, dim=1) if len(supports) != 0 else torch.zeros(b, 1, out_size, device=device)
            masks = torch.cat(masks, dim=1) if len(masks) != 0 else torch.zeros(b, 1, device=device)
            time_modality = torch.cat(t_mods, dim=1) if len(t_mods) != 0 else torch.zeros(b, 1, device=device)

            allow = self.build_allow_mask(time_pivot, time_modality)
            z: torch.Tensor = pivot_out["data"]
            for gated_x_attn in self.gatedXAttn_layers:
                z = gated_x_attn(z, support, attn_mask=allow, q_mask=pivot_out["mask"], kv_mask=masks)

            if self.fusion_pooling is not None:
                z = self.fusion_pooling(z, mask=pivot_out["mask"])

            out.embeddings = z
            return out if not return_dict else asdict(out)

import dataclasses
from dataclasses import asdict
from typing import Optional

import torch
from einops import repeat, rearrange
from torch import nn

from main.model.neegavi.blocks import ModalityStream, ModalContextEncoder, AbstractAttentionBlock
from main.model.neegavi.utils import EegBaseModelOutputs
from main.utils.data import MaskedValue, KdMaskedValue
from main.utils.logging import make_logger


@dataclasses.dataclass
class EegInterAviModelConfiguration:
    # Shapes
    pivot_dim: int
    support_dim: int
    # Configuration variables
    drop_p: float = .0
    use_modality_encoder: bool = True


class EegInterAviModel(nn.Module):
    KD_KEY = "kd"

    def __init__(self, pivot: ModalityStream, *supports: ModalityStream,
                 attn_blocks: list[AbstractAttentionBlock], config: EegInterAviModelConfiguration):
        """

        :param pivot:
        :param supports:
        :param config:
        """
        super(EegInterAviModel, self).__init__()
        self.logger = make_logger(self.__class__.__name__)

        self.pivot: ModalityStream = pivot
        self.supports: nn.ModuleList[ModalityStream] = nn.ModuleList(supports)

        # Declaration beforehand. Check will assign its true value
        self.supports_feature_size: int = -1
        self.check_supports()

        self.drop_p: float = config.drop_p
        self.modality_encoder: Optional[ModalContextEncoder] = None
        if config.use_modality_encoder:
            modality_mappings = {e.get_code(): i for i, e in enumerate(self.supports)}
            self.modality_encoder = ModalContextEncoder(self.supports_feature_size, modality_mappings)
        self.gatedXAttn_layers = nn.ModuleList(attn_blocks)

    def check_supports(self):
        """

        :return:
        """
        if len(self.supports) == 0:
            error_message = "Supports cannot be empty. At least one support must be provided."
            self.logger.error(error_message)
            raise ValueError(error_message)

        self.supports_feature_size: int = self.supports[0].output_size
        base_timestep = self.supports[0].timestep_second
        check_code: str = self.supports[0].code

        for support in self.supports:
            code = support.code
            current_timestep = support.timestep_seconds
            if support.output_size != self.supports_feature_size:
                error_msg = (f"Output size of support {code} ({support.output_size}) does not "
                             f"match extracted size of {check_code} ({self.latent_output_size})")
                self.logger.error(error_msg)
                raise ValueError(error_msg)

            # Assumption of the model is that all supporting modalities are aligned to same timestep size.
            # This can be either true by default or a result of the ModalityStream. We just assume it to be.
            if current_timestep != base_timestep or current_timestep != self.pivot.timestep_seconds:
                msg = f"Timesteps do not match for {code}-{check_code}. Timesteps {current_timestep}!={base_timestep}"
                self.logger.error(msg)
                raise ValueError(msg)

    def rand_select_keep_modality_rows(self, batch_size: int, device, ensure_one: bool = False):
        """

        :param batch_size:
        :param device:
        :param ensure_one:
        :return:
        """
        if (not self.training) or self.drop_p <= 0:
            return torch.ones(batch_size, len(self.supports), device=device)
        keep = torch.bernoulli(torch.full((batch_size, len(self.supports)), 1 - self.drop_p, device=device)).bool()

        dead = ~keep.any(1)
        # TODO decidi se fare ensure di one
        if ensure_one and dead:
            summed_dead = int(dead.sum().item())
            keep[dead, torch.randint(0, len(self.supports), (summed_dead,), device=device)] = True

        return keep

    def process_pivot(self):
        pass

    def build_allow_mask(self, t_q: torch.Tensor, t_kv: torch.Tensor):
        """

        :param t_q:
        :param t_kv:
        :return:
        """
        # By multipliyng by the timestep seconds we reshape so that alignemnt works correctly.
        tq = t_q.unsqueeze(-1) * self.pivot.timestep_seconds  # [B, Tq, 1]
        tk = t_kv.unsqueeze(1) * self.supports[0].timestep_seconds  # [B, 1, Tk]

        if self.allow_modality == "window":
            return (tk - tq).abs() <= self.past_window_units
        if self.allow_modality == "causal":
            return tk <= tq
        raise ValueError(f"Unknown mode: {self.allow_modality}")

    @staticmethod
    def pad_to_batch(data: torch.Tensor, mask: Optional[torch.Tensor], idx: torch.Tensor, b: int):
        """

        :param data:
        :param mask:
        :param idx:
        :param b:
        :return:
        """
        # Pad to same batch size (This happens when we drop some elements from modality)
        # Pad the data
        pad_y = torch.zeros(b, *data.shape[1:], device=data.device)
        pad_y[idx] = data

        # Pad the mask. If non-existent we generate one matching the data tensor.
        if mask is not None:
            pad_mask = torch.zeros(b, *mask.shape[1:], device=mask.device).bool()
            pad_mask[idx] = mask
        else:
            pad_mask = torch.zeros(b, data.size(1), dtype=torch.bool, device=data.device)
            pad_mask[idx] = True

        return MaskedValue(data=pad_y, mask=pad_mask)

    def process_modality(self, x: MaskedValue, idx: torch.Tensor, b: int, modality: ModalityStream, use_kd: bool):
        """

        :param x:
        :param idx:
        :param b:
        :param modality:
        :param use_kd:
        :return:
        """
        output = dict()
        data, mask = x["data"], x.get("mask", None)
        _, t = data.shape[0:2]

        if mask is not None:
            mask = mask.bool()

        y: MaskedValue | KdMaskedValue = modality(data, mask, use_kd=use_kd)
        if self.KD_KEY in y:
            kd = y.pop(self.KD_KEY)
            output[self.KD_KEY] = self.pad_to_batch(kd["data"], kd["mask"], idx, b)

        # We dropped the KD key so the object no longer is a KdMaskedValue if it was
        y: MaskedValue

        z = y["data"]
        if self.modality_encoder is not None:
            z = self.modality_encoder(z, modality=modality.get_code())

        time_mask = torch.arange(t, device=data.device)
        _, _, m, d = z.shape
        # Reshape so that we flatten T x M
        z = rearrange(z, "b t m d -> b (t m) d")
        time_mask = repeat(time_mask, "t -> b (t m)", b=b, m=m)
        if mask is not None:
            mask = repeat(mask, "b t -> b (t m)", m=m)

        res = self.pad_to_batch(z, mask, idx, b)
        return output | {"data": res["data"], "mask": res["mask"], "t_mod": time_mask}

    def forward(self, x: dict, use_kd: bool = False, return_dict: bool = False):
        """

        :param x:
        :param use_kd:
        :param return_dict:
        :return:
        """
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
            mod_data, mod_mask = modality_out["data"], modality_out.get("mask", None)
            out.multimodal_outs[modality_code] = MaskedValue(data=mod_data, mask=mod_mask)

            # For later fusion
            supports.append(mod_data)
            masks.append(mod_mask)
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

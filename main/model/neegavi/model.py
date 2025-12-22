import dataclasses
from abc import ABC, abstractmethod
from dataclasses import asdict
from typing import Optional, Literal, Callable, Any

import torch
from einops import repeat, rearrange
from torch import nn

from main.model.neegavi.blocks import ModalityStream, ModalContextEncoder, AbstractAttentionBlock
from main.model.neegavi.dropout import ModalityDropout, DisabledModalityDropout
from main.model.neegavi.pooling import ClsPooling
from main.model.neegavi.utils import EegBaseModelOutputs, WeaklySupervisedEegBaseModelOutputs
from main.utils.data import MaskedValue, KdMaskedValue
from main.utils.logging import make_logger


@dataclasses.dataclass
class EegInterAviModelConfiguration:
    # Shapes
    pivot_dim: int
    support_dim: int

    output_size: int  # End size of the model (Output).

    # Configuration variables
    drop_p: float = .0
    use_modality_encoder: bool = True


@dataclasses.dataclass
class WeaklySupervisedWrapperModelConfiguration:
    hidden_size: int
    output_size: int


# Utility fn
def check_supports(supports: nn.ModuleList) -> int:
    logger = make_logger(__name__)

    if len(supports) == 0:
        error_message = "Supports cannot be empty. At least one support must be provided."
        logger.error(error_message)
        raise ValueError(error_message)

    reference: ModalityStream = supports[0]
    for support in supports[1:]:
        if support.output_size != reference.output_size:
            error_msg = (
                f"Output size of support {support.code} ({support.output_size}) does not "
                f"match extracted size of {reference.code} ({reference.output_size})"
            )

            logger.error(error_msg)
            raise ValueError(error_msg)

        if support.timestep_seconds != reference.timestep_seconds:
            msg = (
                f"Timesteps do not match for {support.code}-{reference.code}."
                f" Timesteps {support.timestep_seconds}!={reference.timestep_seconds}"
            )

            logger.error(msg)
            raise ValueError(msg)

    return reference.output_size


class EegInterAviModel(nn.Module):
    KD_KEY = "kd"

    def __init__(self,
                 config: EegInterAviModelConfiguration,
                 pivot: ModalityStream, *supports: ModalityStream,
                 modality_dropout: Optional[ModalityDropout] = None,
                 # Attention once modalities are Streamed through their pipeline and cat
                 attn_blocks: list[AbstractAttentionBlock],
                 # Pooling strategy after attention
                 pooling: Optional[nn.Module] = None):
        super(EegInterAviModel, self).__init__()
        self.logger = make_logger(self.__class__.__name__)

        # Pivot defines Q in xattn while supports compose KV
        self.pivot: ModalityStream = pivot
        self.supports: nn.ModuleList[ModalityStream] = nn.ModuleList(supports)

        # Declaration beforehand. Check will assign its true value
        self.supports_feature_size: int = check_supports(self.supports)

        # By default, if not served it is disabled
        if modality_dropout is None:
            modality_dropout = DisabledModalityDropout(len(self.supports))
        self.modality_dropout: ModalityDropout = modality_dropout

        self.modality_encoder: Optional[ModalContextEncoder] = None
        if config.use_modality_encoder:
            modality_mappings = {e.get_code(): i for i, e in enumerate(self.supports)}
            self.modality_encoder = ModalContextEncoder(self.supports_feature_size, modality_mappings)

        # TODO read from config
        self.allow_modality: Literal['window', 'causal'] = 'window'
        self.past_window_units: int = 2  # How much past can be seen

        self.gatedXAttn_layers = nn.ModuleList(attn_blocks)

        # Define Pooling Strategy
        self.cls_idx: int = -1  # Last index is reserved for [CLS] token
        if pooling is None:
            pooling = ClsPooling()
        if isinstance(pooling, ClsPooling) or hasattr(pooling, "cls_idx"):
            pooling.cls_idx = self.cls_idx

        self.pooling: nn.Module = pooling
        self.cls_token = nn.Parameter(torch.randn(1, 1, config.output_size))
        self.use_cls: bool = hasattr(pooling, "cls_idx")  # todo check

    def init_output(self, device):
        empty = torch.zeros(1, device=device)
        return EegBaseModelOutputs(empty, MaskedValue(data=empty, mask=None), {}, {})

    def process_pivot(self, x: MaskedValue, use_kd: bool, out: EegBaseModelOutputs, device) \
            -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        data, mask = x["data"], x.get("mask", None)
        code = self.pivot.get_code()
        b, t = data.shape[0:2]  # Batch and time steps

        time = self.make_pivot_time_map(b, t, device)
        y: MaskedValue | KdMaskedValue = self.pivot(data, mask=mask, use_kd=use_kd)
        if self.KD_KEY in y:
            out.kd_outs[code] = y.pop(self.KD_KEY)
        out.multimodal_outs[code] = y
        return y["data"], y["mask"], time

    def make_pivot_time_map(self, b: int, t: int, device) -> torch.Tensor:
        return repeat(torch.arange(t, device=device), "t -> b t", b=b)

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

    def build_allow_mask(self, t_q: torch.Tensor, t_kv: torch.Tensor):
        """
        Allowance mask aligns the same timesteps and previous ones.
        If modality is window only a limited amount of past is preserved as context.
        :param t_q: Time map for query vector
        :param t_kv: Time map for kv vectory
        :return: Allowance mask signaling what one can attend to.
        """
        # By multiplying by the timestep seconds we reshape so that alignemnt works correctly.
        tq = t_q.unsqueeze(-1) * self.pivot.timestep_seconds  # [B, Tq, 1]
        tk = t_kv.unsqueeze(1) * self.supports[0].timestep_seconds  # [B, 1, Tk]
        if self.allow_modality == "window":
            # Window size compels us to explode the sun.
            # It's written like this to avoid symmetry
            return (tq >= (tk - self.past_window_units)) & (tq < tk + self.supports[0].timestep_seconds)

        if self.allow_modality == "causal":
            return tk <= tq
        raise ValueError(f"Unknown mode: {self.allow_modality}")

    def process_support(self, x: MaskedValue, keep_idx: torch.Tensor, modality: ModalityStream,
                        use_kd: bool, out: EegBaseModelOutputs, device):
        data, mask = x["data"], x.get("mask", None)
        b, t = data.shape[0:2]
        code = modality.get_code()

        if mask is not None:
            mask = mask.bool()
        y: MaskedValue | KdMaskedValue = modality(data, mask, use_kd=use_kd)

        if self.KD_KEY in y:
            kd_out: MaskedValue = y.pop(self.KD_KEY)
            out.kd_outs[code] = self.pad_to_batch(kd_out["data"], kd_out["mask"], keep_idx, b)

        y: MaskedValue
        z, _ = y["data"], y["mask"]

        if self.modality_encoder is not None:
            z = self.modality_encoder(z, modality=code)

        _, _, m, d = z.shape
        z = rearrange(z, "b t m d -> b (t m) d")
        time = torch.arange(t, device=data.device)
        time = repeat(time, "t -> b (t m)", b=b, m=m)

        if mask is not None:
            mask = repeat(mask, "b t -> b (t m)", m=m)

        z = self.pad_to_batch(z, mask, keep_idx, b)

        out.multimodal_outs[code] = z
        return z["data"], z["mask"], time

    def process_supports(self, x: dict, keep: torch.Tensor, use_kd: bool, out: EegBaseModelOutputs, device):
        b = keep.shape[0]  # TODO check

        # Empty initialization
        support = torch.empty(b, 0, self.supports_feature_size, device=device)
        mask = torch.empty(b, 0, device=device, dtype=torch.bool)
        time = torch.empty(b, 0, device=device)

        modality: ModalityStream
        for idx, modality in enumerate(self.supports):
            keep_idx = keep[:, idx].nonzero(as_tuple=True)[0]

            support_out, support_mask, support_time = self.process_support(
                x=x[modality.get_code()], keep_idx=keep_idx, modality=modality, use_kd=use_kd, out=out, device=device
            )

            # Add the found elements
            support = torch.cat([support, support_out], dim=1)
            mask = torch.cat([mask, support_mask], dim=1)
            time = torch.cat([time, support_time], dim=1)

        return support, mask, time

    def forward(self, x: dict, use_kd: bool = False, return_dict: bool = False):
        device = x[self.pivot.get_code()]["data"].device
        out = self.init_output(device=device)

        pivot_out, pivot_mask, pivot_time = self.process_pivot(
            x[self.pivot.get_code()], use_kd=use_kd, out=out, device=device
        )

        b = pivot_out.shape[0]
        support_out, support_mask, support_time = self.process_supports(
            x, self.modality_dropout(b, device), use_kd, out, device
        )

        q, q_mask = pivot_out, pivot_mask

        q = torch.cat([q, self.cls_token.expand(q.shape[0], -1, -1)], dim=1)
        cls_mask = torch.ones(pivot_mask.shape[0], 1, device=q.device, dtype=q_mask.dtype)
        if not self.use_cls:
            cls_mask = torch.zeros(pivot_mask.shape[0], 1, device=q.device, dtype=q_mask.dtype)
        q_mask = torch.cat([q_mask, cls_mask], dim=1)

        cls_time = pivot_time.new_full((b, 1), pivot_time.size(1))
        pivot_time = torch.cat([pivot_time, cls_time], dim=1)  # length T+1

        allow = self.build_allow_mask(pivot_time, support_time)
        allow[:, -1, :] = True

        for xattn in self.gatedXAttn_layers:
            q = xattn(q, support_out, attn_mask=allow, q_mask=q_mask.bool(), kv_mask=support_mask)

        out.embeddings = MaskedValue(data=q, mask=q_mask)
        out.cls = self.pooling(q, q_mask)
        return out


class OldEegInterAviModel(nn.Module):
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

        self.allow_modality: Literal['window', 'causal'] = 'window'
        self.past_window_units: int = 2  # How much past can be seen

        self.drop_p: float = config.drop_p
        self.modality_encoder: Optional[ModalContextEncoder] = None
        if config.use_modality_encoder:
            modality_mappings = {e.get_code(): i for i, e in enumerate(self.supports)}
            self.modality_encoder = ModalContextEncoder(self.supports_feature_size, modality_mappings)

        self.gatedXAttn_layers = nn.ModuleList(attn_blocks)

        # self.fusion_pooling = MaskedPooling()
        self.fusion_pooling = None

        self.config = config

        self.cls_token: Optional[nn.Parameter] = None
        if isinstance(token_pooling, ClsPooling):
            self.cls_token = nn.Parameter(torch.randn(1, 1, pivot.output_size))

    def check_supports(self):
        """

        :return:
        """
        if len(self.supports) == 0:
            error_message = "Supports cannot be empty. At least one support must be provided."
            self.logger.error(error_message)
            raise ValueError(error_message)

        self.supports_feature_size: int = self.supports[0].output_size
        base_timestep = self.supports[0].timestep_seconds
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
            if current_timestep != base_timestep:
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

    def build_allow_mask(self, t_q: torch.Tensor, t_kv: torch.Tensor):
        """
        Allowance mask aligns the same timesteps and previous ones.
        If modality is window only a limited amount of past is preserved as context.
        :param t_q: Time map for query vector
        :param t_kv: Time map for kv vectory
        :return: Allowance mask signaling what one can attend to.
        """
        # By multiplying by the timestep seconds we reshape so that alignemnt works correctly.
        tq = t_q.unsqueeze(-1) * self.pivot.timestep_seconds  # [B, Tq, 1]
        tk = t_kv.unsqueeze(1) * self.supports[0].timestep_seconds  # [B, 1, Tk]
        if self.allow_modality == "window":
            # Window size compels us to explode the sun.
            # It's written like this to avoid symmetry
            return (tq >= (tk - self.past_window_units)) & (tq < tk + self.supports[0].timestep_seconds)

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

        _, _, m, d = z.shape
        # Reshape so that we flatten T x M
        z = rearrange(z, "b t m d -> b (t m) d")
        time_mask = torch.arange(t, device=data.device)
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
        out = EegBaseModelOutputs(torch.zeros(1), None, {}, {})

        # Pivot modality elaboration
        pivot_x = x[self.pivot.get_code()]
        pivot_data, pivot_mask = pivot_x["data"], pivot_x.get("mask", None)
        # Device to always use the same
        device = pivot_data.device
        b, t = pivot_data.shape[0:2]

        # Time of pivot
        # For CLS, we have to add 1
        time_range = t + 1
        time_pivot = torch.arange(time_range, device=device)
        time_pivot = repeat(time_pivot, "t -> b t", b=b)

        pivot_out = self.pivot(pivot_data, mask=pivot_mask, use_kd=use_kd)
        if self.KD_KEY in pivot_out:
            out.kd_outs[self.pivot.get_code()] = pivot_out.pop(self.KD_KEY)
        out.multimodal_outs[self.pivot.get_code()] = pivot_out

        # Supporting modalities elaboration
        keep = self.rand_select_keep_modality_rows(b, device)
        supports, masks, t_mods = [], [], []
        modality: ModalityStream
        for idx, modality in enumerate(self.supports):
            modality_code = modality.get_code()
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

        cls_index = -1  # TODO vedi se questo va a guardare step non validi
        allow = self.build_allow_mask(time_pivot, time_modality)
        allow[:, cls_index, :] = True

        z: torch.Tensor = pivot_out["data"]
        # Add the CLS token
        cls = self.cls_token.expand(z.shape[0], -1, -1)
        z = torch.cat((z, cls), dim=1)
        cls_mask = torch.ones(pivot_mask.shape[0], 1, device=z.device, dtype=pivot_mask.dtype)
        pivot_mask = pivot_mask.any(dim=-1)
        pivot_mask = torch.cat((pivot_mask, cls_mask), dim=1)

        for gated_x_attn in self.gatedXAttn_layers:
            z = gated_x_attn(z, support, attn_mask=allow, q_mask=pivot_mask.bool(), kv_mask=masks)

        # Because the mask was on channels, but we got rid of them
        out.embeddings = MaskedValue(data=z, mask=pivot_mask)
        out.cls = self.get_cls(z, pivot_mask)

        return out if not return_dict else asdict(out)

    def get_cls(self, z: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # mask = mask.unsqueeze(-1).float()
        # return (z * mask).sum(dim=1) / mask.sum(dim=1)
        return z[:, -1]  # Class is last


class WeaklySupervisedEegInterAviModel(nn.Module):
    def __init__(self, base_model: EegInterAviModel, base_model_out_size: int, hidden_size: int, output_size: int):
        super().__init__()
        self.base_model = base_model
        self.prediction_head = nn.Sequential(
            nn.Linear(base_model_out_size, hidden_size),
            nn.ReLU(),
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, x: dict, use_kd: bool = False, return_dict: bool = False):
        outs: EegBaseModelOutputs = self.base_model(x, use_kd=use_kd, return_dict=False)
        pred = self.prediction_head(outs.embeddings)
        o = WeaklySupervisedEegBaseModelOutputs(pred=pred, **vars(outs))
        return o if not return_dict else asdict(o)

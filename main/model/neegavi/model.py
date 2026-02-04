import dataclasses
from dataclasses import asdict
from typing import Optional, Literal

import torch
from einops import repeat, rearrange
from torch import nn
from torch.distributed import supports_complex

from main.model.neegavi.blocks import ModalityStream, ModalContextEncoder, AbstractAttentionBlock, TimeMaskSwitchable, \
    TimeMaskSwitchableProperties
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
    modality: TimeMaskSwitchableProperties

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


# todo refactorino per capire meglio
class EegInterAviModel(nn.Module, TimeMaskSwitchable):
    KD_KEY = "kd"

    def __init__(self,
                 config: EegInterAviModelConfiguration,
                 pivot: ModalityStream, *supports: ModalityStream,
                 modality_dropout: Optional[ModalityDropout] = None,
                 attn_blocks: list[AbstractAttentionBlock],
                 pooling: Optional[nn.Module] = None):
        """
        EegInterVaiModel is partially inspired by the novel approach of the Flamingo model by Google.
        It keeps the same idea of interleaving different modality data but extends it on the time axis.
        This is because the data we analise has a strong temporal relationship (it evolves as the measurement goes on).

        The broad idea is to do xattn to enrich and hopefully to fuse information of multiple modalities into a pivot (EEG).
        This idea is achieved by the same way it was done in Flamingo via gatedxattn.


        Broad schema of the workflow:
        - modality -> reshape -> adapter (Perceiver Resampler, Feed Forward) -> OUT
                                                                             -> KD head -x (branch dies here)
        then modalities are collected: p=pivot_OUT, s=cat([supports_OUT])
        out = xattn(q=p, kv=s)

        :param config:
        :param pivot: Main modality to fuse data into. It is Q in the late xattn part of the model.
        :param supports: They build KV in xattn and are modalities that should enrich the pivot
        :param modality_dropout:
        :param attn_blocks: Attention once modalities are Streamed through their pipeline and cat
        :param pooling: Pooling strategy after attention. Model has a [CLS] token thus it is not needed. @deprecated
        """
        nn.Module.__init__(self)
        # The model operates with time steps so it has its own logic what masking concerns
        TimeMaskSwitchable.__init__(self)

        self.logger = make_logger(self.__class__.__name__)
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

        self.set_attention_modality(config.modality)
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

    @staticmethod
    def init_output(device):
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

    @staticmethod
    def make_pivot_time_map(b: int, t: int, device) -> torch.Tensor:
        return repeat(torch.arange(t, device=device), "t -> b t", b=b)

    @staticmethod
    def pad_to_batch(data: torch.Tensor, mask: Optional[torch.Tensor], idx: torch.Tensor, b: int):
        """
        This utility function play an important role the moment we drop some modalities.
        That procedure breaks batch as some elements go missing for some modalities, this enures to restore with empty objects
        the original batch structure.
        :param data: Data to pad to batch_size
        :param mask: Mask to pad back to batch_size
        :param idx: Indexes that were lost previously. Without this padding would be impossible as we need to restore in
        the correct places to evaluate loss and other metrics. (Siglip would of course break as I'd be comparing wrong samples)
        :param b: the batch_size
        :return: Restored MaskedValue of the original batch shape
        """
        # Pad to same batch size (This happens when we drop some elements from modality)
        # Pad the data
        pad_y = torch.zeros(b, *data.shape[1:], device=data.device, dtype=data.dtype)
        pad_y[idx] = data

        # Pad the mask. If non-existent we generate one matching the data tensor.
        if mask is not None:
            pad_mask = torch.zeros(b, *mask.shape[1:], device=mask.device, dtype=torch.bool)
            pad_mask[idx] = mask
        else:
            pad_mask = torch.zeros(b, data.size(1), device=data.device, dtype=torch.bool)
            pad_mask[idx] = True

        return MaskedValue(data=pad_y, mask=pad_mask)

    def build_allow_mask(self, t_q: torch.Tensor, t_kv: torch.Tensor):
        """
        TODO: Could this be cached by configuration?
        Allowance mask aligns the same timesteps and previous ones.
        If modality is window only a limited amount of past is preserved as context.
        :param t_q: Time map for query vector
        :param t_kv: Time map for kv vectory
        :return: Allowance mask signaling what one can attend to.
        """
        # By multiplying by the timestep seconds we reshape so that alignemnt works correctly.
        dt = self.supports[0].timestep_seconds
        tq = t_q.unsqueeze(-1) * self.pivot.timestep_seconds  # [B, Tq, 1]
        tk = t_kv.unsqueeze(1) * dt  # [B, 1, Tk]

        if self.modality.mode == "window":
            lb, la = self.modality.lookback, self.modality.lookahead
            # Window size compels us to explode the sun.
            # It's written like this to avoid symmetry
            return (tq >= (tk - lb * dt)) & (tq < (tk + dt + la * dt))

        if self.modality.mode == "causal":
            return tq < tk + dt

        if self.modality.mode == "bidirectional":
            return torch.ones(tq.shape[0], tq.shape[1], tk.shape[-1], device=tq.device, dtype=torch.bool)

        raise ValueError(f"Unknown mode: {self.modality.mode}")

    def process_support(self, x: MaskedValue, keep_idx: torch.Tensor, modality: ModalityStream,
                        use_kd: bool, out: EegBaseModelOutputs):
        data, mask = x["data"], x.get("mask", None)
        b, t = data.shape[0:2]

        if mask is not None:
            mask = mask.bool()

        y: MaskedValue | KdMaskedValue = modality(data[keep_idx], mask[keep_idx], use_kd=use_kd)
        if self.KD_KEY in y:
            kd_out: MaskedValue = y.pop(self.KD_KEY)
            out.kd_outs[modality.get_code()] = self.pad_to_batch(kd_out["data"], kd_out["mask"], keep_idx, b)

        y: MaskedValue
        z, _ = y["data"], y["mask"]

        if self.modality_encoder is not None:
            z = self.modality_encoder(z, modality=modality.get_code())

        _, _, m, d = z.shape
        z = rearrange(z, "b t m d -> b (t m) d")
        time = torch.arange(t, device=data.device)
        time = repeat(time, "t -> b (t m)", b=b, m=m)

        if mask is not None:
            mask = repeat(mask, "b t -> b (t m)", m=m)

        z = self.pad_to_batch(z, mask[keep_idx], keep_idx, b)
        out.multimodal_outs[modality.get_code()] = z
        return z["data"], z["mask"], time

    def process_supports(self, x: dict, keep: torch.Tensor, use_kd: bool, out: EegBaseModelOutputs, device, dtype):
        b = keep.shape[0]
        # Empty initialization
        support_outs, mask_outs, time_outs = [], [], []
        modality: ModalityStream
        for idx, modality in enumerate(self.supports):
            if modality.get_code() not in x:
                continue
            # keep index map of the current modality
            keep_idx = keep[:, idx].nonzero(as_tuple=True)[0]
            # Skip if empty
            if keep_idx.numel() == 0:
                continue

            support_out, support_mask, support_time = self.process_support(
                x=x[modality.get_code()], keep_idx=keep_idx, modality=modality, use_kd=use_kd, out=out
            )

            # Add the found elements
            support_outs.append(support_out.to(dtype))
            mask_outs.append(support_mask.bool())
            time_outs.append(support_time)

        if len(support_outs) != 0:
            # Filled return objects
            support = torch.cat(support_outs, dim=1)
            mask = torch.cat(mask_outs, dim=1)
            time = torch.cat(time_outs, dim=1)
        else:
            # Default empty return
            support = torch.empty(b, 0, self.supports_feature_size, device=device)
            mask = torch.empty(b, 0, device=device, dtype=torch.bool)
            time = torch.empty(b, 0, device=device)

        return support, mask, time

    def forward(self, x: dict, use_kd: bool = False, return_dict: bool = False):
        # Initialize current device and output object
        device = x[self.pivot.get_code()]["data"].device
        out = self.init_output(device=device)

        # Process the pivot before fusion via xattn
        pivot_out, pivot_mask, pivot_time = self.process_pivot(
            x[self.pivot.get_code()], use_kd=use_kd, out=out, device=device
        )

        b = pivot_out.shape[0]
        # Process the supports before fusion via xattn
        support_out, support_mask, support_time = self.process_supports(
            x, self.modality_dropout(b, device), use_kd, out, device, pivot_out.dtype
        )

        # Add CLS token and adapt masks
        q, q_mask = pivot_out, pivot_mask
        q = torch.cat([q, self.cls_token.expand(q.shape[0], -1, -1)], dim=1)
        cls_mask = torch.ones(pivot_mask.shape[0], 1, device=q.device, dtype=q_mask.dtype)
        if not self.use_cls:
            cls_mask = torch.zeros(pivot_mask.shape[0], 1, device=q.device, dtype=q_mask.dtype)
        q_mask = torch.cat([q_mask, cls_mask], dim=1)

        cls_time = pivot_time.new_full((b, 1), pivot_time.size(1))
        pivot_time = torch.cat([pivot_time, cls_time], dim=1)  # length T+1

        # Build the mask that maps visibility of timesteps to each other.
        # A timestep i can maybe see j < i but not any j > i (This would mean only past). Strategy is defined on upper level,
        # but it can change at runtime thus the build allowance maks can change during training.
        allow = self.build_allow_mask(pivot_time, support_time)
        allow[:, -1, :] = True

        for xattn in self.gatedXAttn_layers:
            q = xattn(q, support_out, attn_mask=allow, q_mask=q_mask.bool(), kv_mask=support_mask)

        out.embeddings = MaskedValue(data=q, mask=q_mask)
        out.cls = self.pooling(q, q_mask)
        return out

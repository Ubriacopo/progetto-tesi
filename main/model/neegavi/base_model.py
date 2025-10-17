"""

    embedders are frozen. I get input embeddings already
    pivot is EEG rest are supporting modalities.s

    EEG shape: [b, Te, c, De]
    EEG mask: [b, Te]

    Aud shape: [b, Ta, Pa, Da]
    Aud mask: [b, Ta]

    Vid shape: [b, Tv, Pv, Dv]
    Vid mask: [b, Tv]

    Step 1 : Ridurre la dimensione di Pa e Pv per comprimere informazioni maggiormente.
    (POOLING) & (All support to same M) (Attn pooling o altro)
    Vid [b, Tv, Pv, Dv] -> [b, Tv, M, Dv] (M << Pv)
    Aud [b, Ta, Pa, Da] -> [b, Ta, M, Da]

    Step 2: Remap to same D (supports)
    Vid [b, Tv, M, Dv] -> [b, Tv, M, D]
    Aud [b, Ta, M, Da] -> [b, Ta, M, D]

    (Ma prima step 1 o step 2?)

    Step 3: Costruire t_vid, t_aud
    t_vid = [b, T] -> Quindi se T = 11 e ogni T dura 3 s -> [0, 3, 6, 9, 12, 15 ...]
    t_aud = [b, T] -> Quindi se T = 34 e ogni T dura 0.96s -> [0, 0.96, 1.92 ...]

    Step 4: Flatten M & T per ogni modality. Quindi cambiare anche la sua maschera e t_object
    Vid [b, Tv, M, D] -> [b, Tv*M, D]
    t_vid [b, Tv] -> [b, Tv*M]
    vid_mask [b, Tv] -> [b, Tv*M]
    Esempio se t_mask= [1,1,0...] (solo primi 6 secondi validi) e M= 2 allora:
    vid_mask = [1,1,1,1,0...0]

    A questo punto posso costruire kv come:
    kv = torch.cat([Vid, Aud]
    t_kv = torch.cat([t_vid, t_aud])
    mask_kv= torch.cat([vid_mask, aud_mask])

    Ora posso fare xattn con q=EEG, kv=kv
    Devo passare a xattn anche mask_kv, mask_EEG, e t_kv e t_eeg
    Con questi dati posso assicurami che (ad esempio)

    - Se EEG valido a tempo fino al T=4 (4 secondi) (mask=[1,1,1,1,0....] prima di espandere su M)
        Devo fare attn su t_vid su tutto t_vid fino a 4 (primi 6 secondi visto che 4 >3 e devo prendere due step)
        Allora posso fare attn su t_aud quando t_aud > 3 e t_aud <4 (qui circa)

        Dovrei anche vedere cose del passato?
        Essendo un EEG ha senso che quello che provo deriva da tempi passati?
        Non diventa troppo il contesto? Potrei usare una window...
        Ho forse mal compreso come interagiscono i dati.

    Attn part:

"""
import logging
import math
from dataclasses import asdict
from typing import Optional, Literal

import torch
from einops import repeat, rearrange
from torch import nn

from main.model.EEGAVI.base_model import EegBaseModelOutputs, FusionPooling, WeaklySupervisedEegBaseModelOutputs
from main.model.neegavi.xattention import GatedXAttentionCustomArgs, GatedXAttentionBlock
from main.model.layer.base import ModalContextEncoder
from main.model.layer.modality_stream import ModalityStream
from main.utils.data import MaskedValue, KdMaskedValue


class NEEGAviModel(nn.Module):
    def __init__(self, output_size: int,
                 pivot: ModalityStream, supports: list[ModalityStream],
                 xattn_blocks: int | list[GatedXAttentionCustomArgs],
                 drop_p: float = 0.0, use_modality_encoder: bool = True,
                 max_sequence_seconds: int = 32, ):
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

        self.max_sequence_seconds: int = max_sequence_seconds
        self.drop_p = drop_p
        self.fused_norm = nn.LayerNorm(output_size)
        self.allow_modality: Literal['window', 'causal'] = 'window'
        self.window_seconds = 2
        self.fusion_pooling: nn.Module = self.build_fusion_pooling()

    def build_fusion_pooling(self):
        return FusionPooling()

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

        y["data"] = z

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
        # todo turn into mini model this part so it is not part of neegavi def
        for idx, modality in enumerate(self.supports):
            modality: ModalityStream
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

        z = self.fusion_pooling(z, mask=pivot_output["mask"])
        z = self.fused_norm(z)
        return_object = EegBaseModelOutputs(embeddings=z, kd_outs=kd_outputs, multimodal_outs=multimodal_outs)
        return return_object if not return_dict else asdict(return_object)


class WeaklySupervisedNEEEGBaseModel(nn.Module):
    def __init__(self, eeg_base_model: NEEGAviModel, hidden_size: int, output_size: int):
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

import dataclasses
from dataclasses import asdict
from traceback import print_stack
from typing import Optional

import torch
from torch import nn

from main.model.EEGAVI.utils import remap_with_overlap
from main.model.layer.attention.x_attention import GatedXAttentionBlock, GatedXAttentionCustomArgs
from main.model.layer.base import ModalContextEncoder
from main.model.layer.modality_stream import ModalityStream
from main.utils.data import MaskedValue


@dataclasses.dataclass
class EEGAVIOutputs:
    embeddings: torch.Tensor
    kd_outs: dict[str, MaskedValue]
    multimodal_outs: dict[str, MaskedValue]


class EEGAVI(nn.Module):
    def __init__(self,
                 pivot_latent_size: int, pivot_modality: ModalityStream,
                 supporting_latent_size: int, supporting_modalities: list[ModalityStream],
                 xattn_blocks: int | list[GatedXAttentionCustomArgs],
                 remap_timesteps: int,
                 use_modality_encoder: bool = True,
                 drop_p: float = 0.1
                 ):
        """
        EEGAVI behaves like Flamingo models. It has a main feature space that acts as query in the attn mechanism.
        We call this one pivot and holds a special spot in the architecture as it is the main modality.
        Supporting modalities are modalities that enrich the embeddings of the starting space.
        Before running the attention they are adapted according to their adaption stream and later merged to act as
        query and values of the attn mechanism.

        :param pivot_latent_size:
        :param pivot_modality:
        :param supporting_latent_size:
        :param supporting_modalities:
        :param xattn_blocks: int if you want to use default configuration else GatedXAttentionCustomArgs for customization
        :param remap_timesteps:
        :param use_modality_encoder:
        :param drop_p:
        """
        super(EEGAVI, self).__init__()

        self.pivot_latent_size: int = pivot_latent_size
        self.supporting_latent_size: int = supporting_latent_size

        self.pivot_modality: ModalityStream = pivot_modality
        self.supporting_modalities = nn.ModuleList(supporting_modalities)
        self.modality_encoder: Optional[ModalContextEncoder] = None

        if use_modality_encoder:
            modality_mappings = {e.get_code(): i for i, e in enumerate(supporting_modalities)}
            self.modality_encoder = ModalContextEncoder(supporting_latent_size, modality_mappings)

        self.gatedXAttn_layers = nn.ModuleList(self.build_xattn_blocks(xattn_blocks))
        self.norm = nn.LayerNorm(pivot_latent_size)
        self.remap_timesteps: int = remap_timesteps
        self.drop_p: float = drop_p

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

    @staticmethod
    def remask(supp: torch.Tensor, device):
        b, T, F, D = supp.shape
        key_time_idx = torch.arange(T, device=device).repeat_interleave(F)
        # allow[q_t, k] = (time(k) <= q_t)
        allow = key_time_idx.view(1, 1, -1) <= torch.arange(T, device=device).view(1, T, 1)
        return allow

    def select_keeps(self, b: int, device):
        """
        Choose what modalities to keep and what to drop (dropout). At least one is always used.
        Selection is single sample based. This means that for each sample in batch every modality can be either on or off.


        :param b: Batch size of the current input sample.
        :return: bool tensor that specifies what modalities are kept and what are dropped.
        """
        n_modalities = len(self.supporting_modalities)
        if (not self.training) or self.drop_p <= 0:
            # In this case everything is kept
            return torch.ones(b, n_modalities, dtype=torch.bool, device=device)

        keep = torch.bernoulli(torch.full((b, n_modalities), 1 - self.drop_p, device=device)).bool()
        dead = ~keep.any(1)
        if dead.any():
            # We force at least one modality to always be on.
            keep[dead, torch.randint(0, n_modalities, (dead.sum(),), device=device)] = True

        return keep
    # TODO vedi di semplificare sta roba
    def forward(self, x: dict, use_kd: bool = False, return_dict: bool = False) \
            -> EEGAVIOutputs | dict[str, MaskedValue]:
        kd_outs: dict = {}
        multimodal_outputs: dict = {}

        # First work with the base modality. (EEG in our case)
        key: str = self.pivot_modality.get_code()
        base, base_mask = x[key]["data"], x[key]["mask"] if "mask" in x[key] else None
        base = self.pivot_modality(base, mask=base_mask, use_kd=use_kd)
        if isinstance(base, tuple):
            # Store the KD output to return later
            kd_outs[key] = base[1]
            # Now we can really get the resampled embeddings
            base = base[0]
        if isinstance(base, dict):
            base, base_mask = base["data"], base["mask"]

        multimodal_outputs[key] = {"data": base, "mask": base_mask}

        supports: list[torch.Tensor] = []
        masks: list[torch.Tensor] = []

        b = next(iter(x.values()))["data"].shape[0]
        keep = self.select_keeps(b, base.device)
        for m, adapter in enumerate(self.supporting_modalities):
            key: str = adapter.get_code()
            idx = keep[:, m].nonzero(as_tuple=True)[0]
            if idx.numel() == 0: continue  # If None taken we skip entirely

            supp, mask = x[key]["data"], x[key].get("mask", None)
            adapted_supp = adapter(supp[idx], mask=mask[idx] if mask is not None else None, use_kd=use_kd)
            if isinstance(adapted_supp, tuple):
                # Store the KD output to return later
                kd = adapted_supp[1]
                try:
                    restored_kd = torch.zeros(supp.shape[0], *kd["data"].shape[1:], device=supp.device)
                    restored_kd[idx] = kd["data"]
                    restored_mask = torch.zeros(supp.shape[0], *kd["mask"].shape[1:], device=supp.device).bool()
                    restored_mask[idx] = kd["mask"]

                    kd_outs[key] = {"data": restored_kd, "mask": restored_mask}
                    # Now we can really get the embeddings
                    adapted_supp = adapted_supp[0]
                except Exception as e:
                    print_stack()
                    raise e

            if isinstance(adapted_supp, dict):
                adapted_supp, _ = adapted_supp["data"], adapted_supp["mask"]

            # Modality embedding if wanted
            if self.modality_encoder is not None:
                adapted_supp = self.modality_encoder(adapted_supp, modality=key)

            Y = adapted_supp.new_zeros(b, *adapted_supp.shape[1:])
            Y[idx] = adapted_supp
            if mask is None:
                M = torch.zeros(b, Y.size(1), dtype=torch.bool, device=Y.device)
                M[idx] = True
            else:
                M = mask.new_zeros(b, Y.size(1))
                M[idx] = mask[idx]

            # Reshape to same size of timesteps
            Y, M = remap_with_overlap(Y, M, self.remap_timesteps)
            Y *= M[:, :, None, None]
            M = M[:, :, None].expand(-1, -1, Y.shape[2])

            masks.append(M)
            supports.append(Y)
            # For output (Loss calculation).
            multimodal_outputs[key] = {"data": Y, "mask": M}

        supp = torch.cat(supports, dim=2)
        supp_mask = torch.cat(masks, dim=2)
        # Prepare attention mask (What is seeable)
        allow = self.remask(supp=supp, device=supp.device)

        # Initialize the variable to anything
        z: torch.Tensor = base
        for gated_x_attn in self.gatedXAttn_layers:
            z = gated_x_attn(z, supp, attn_mask=allow, q_mask=base_mask, kv_mask=supp_mask)

        # TODO: for now simple pooling we could use some learned pooling later on
        w = base_mask.float().sum(dim=-1, keepdim=True).clamp_min(1e-6)
        z = (z * base_mask.unsqueeze(-1)).sum(dim=-2) / w  # Normalization factor
        z = self.norm(z)

        return_object = EEGAVIOutputs(embeddings=z, kd_outs=kd_outs, multimodal_outs=multimodal_outputs)
        return return_object if not return_dict else asdict(return_object)


@dataclasses.dataclass
class WeaklySupervisedEEGAVIOutputs(EEGAVIOutputs):
    pred: torch.Tensor


class WeaklySupervisedEEGAVI(nn.Module):
    def __init__(self, eeg_avi: EEGAVI, hidden_size: int, supervised_target_size: int = 4):
        super().__init__()
        self.eeg_avi: EEGAVI = eeg_avi
        self.prediction_head = nn.Sequential(
            nn.Linear(eeg_avi.pivot_latent_size, hidden_size),
            nn.ReLU(),
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, supervised_target_size)
        )

    def forward(self, x: dict, use_kd: bool = False, return_dict: bool = False):
        outs: EEGAVIOutputs = self.eeg_avi(x, use_kd=use_kd, return_dict=False)
        pred = self.prediction_head(outs.embeddings)
        o = WeaklySupervisedEEGAVIOutputs(pred=pred, **asdict(outs))
        return o if not return_dict else asdict(o)

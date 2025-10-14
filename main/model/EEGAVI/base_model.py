import torch
from torch import nn

from main.model.layer.modality_stream import ModalityStream
from main.utils.data import MaskedValue


class EegBaseModel(nn.Module):
    def __init__(self, pivot: ModalityStream, supports: list[ModalityStream], drop_p: float = 0.0):
        super().__init__()
        self.pivot = pivot
        self.supports = supports

        self.drop_p = drop_p

    def process_pivot(self, x: MaskedValue, use_kd: bool):
        data = x["data"]
        mask = x.get("mask", None)

        y = self.pivot_modality(data, mask=mask, use_kd=use_kd)

        kd_out = None
        if isinstance(y, tuple):
            kd_out, y = y
        if isinstance(y, dict):
            y, mask = y["data"], y["mask"] if "mask" in y else None

        return {"data": y, "mask": mask, "kd_out": kd_out, }

    def process_modality(self, x: MaskedValue, idx: torch.Tensor, modality: ModalityStream, use_kd: bool):
        data = x["data"]
        mask = x.get("mask", None)

        y = modality(data[idx], mask=mask[idx] if mask is not None else None, use_kd=use_kd)
        if isinstance(y, tuple):
            pass

    def forward(self, x: dict, use_kd: bool = False, return_dict: bool = False) -> torch.Tensor:
        kd_outs: dict = {}

        device = x[self.pivot.get_code()]["data"].device

        pivot_out = self.process_pivot(x[self.pivot.get_code()], use_kd=use_kd)
        if pivot_out["kd_out"] is not None:
            kd_outs[self.pivot.get_code()] = pivot_out["kd_out"]

        B = x[self.pivot.get_code()]["data"].shape[0]  # Batch size
        keep = self.select_keeps(B, device)

        supports = []
        masks = []

        for modality_idx, modality in enumerate(self.supports):
            idx = keep[:, modality_idx].nonzero(as_tuple=True)[0]
            self.process_modality(x[modality.get_code()], idx=idx, modality=modality, use_kd=use_kd)

        support = torch.cat(supports, dim=2)
        masks = torch.cat(masks, dim=2)

        allow = self.remask(supp=support, device=support.device)
        z: torch.Tensor = pivot_out["data"]

    def select_keeps(self, b: int, device):
        number_modalities = len(self.supports)
        if (not self.training) or self.drop_p <= 0:
            return torch.ones(b, number_modalities, device=device)

        keep = torch.bernoulli(torch.full((b, number_modalities), 1 - self.drop_p, device=device)).bool()
        dead = ~keep.any(1)
        if dead.any():
            # We force at least one modality to always be on.
            keep[dead, torch.randint(0, number_modalities, (dead.sum(),), device=device)] = True

        return keep

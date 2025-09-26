from __future__ import annotations

import dataclasses
from copy import deepcopy
from typing import List, Optional

import requests
import torch
from jedi.inference.gradual.typing import Callable
from torch import nn

from common.data.ecg.ecg import ECG
from common.data.transform import IDENTITY
from common.data.utils import sanitize_for_ast, timed


@dataclasses.dataclass
class ECGPayload:
    signal: List[List[float]]  # ECG signal data: [leads, samples], e.g., [12, 5000]
    fs: Optional[int] = None  # Sampling rate in Hz (default: 500)
    # TODO: posso estrarli
    patient_age: Optional[int] = None  # Patient age in years
    patient_gender: Optional[str] = None  # Patient gender (M/F)"
    # TODO capire come estrarlik
    lead_names: Optional[List[str]] = None  # Lead names (default: 12-lead standard)

    @staticmethod
    def from_ecg(ecg: ECG, data_transform_fn: Callable[[ECG], ECG]) -> ECGPayload:
        ecg = data_transform_fn(ecg)
        # For serialization
        return ECGPayload(
            signal=ecg.data.tolist(),
            fs=ecg.fs,
            patient_age=ecg.patient_age,
            patient_gender=ecg.patient_gender,
            lead_names=ecg.leads
        )


class EcgFmEmbedderTransform(nn.Module):
    def __init__(self, data_transform_fn: Callable[[ECG], ECG], endpoint: str = "localhost:7860/extract_features"):
        super(EcgFmEmbedderTransform, self).__init__()
        self.endpoint: str = endpoint
        self.data_transform_fn = data_transform_fn

    @timed()
    def forward(self, x: ECG):
        payload = ECGPayload.from_ecg(x, self.data_transform_fn)
        obj = dataclasses.asdict(payload)
        obj = sanitize_for_ast(obj)

        embeddings: Optional[torch.Tensor] = None
        for i in obj["signal"]:
            iter_obj = deepcopy(obj)
            iter_obj["signal"] = i

            r = requests.post(f"http://{self.endpoint}", json=iter_obj)
            features = r.json()["features"]["values"]
            features = torch.tensor(features).float()
            embeddings = features if embeddings is None else torch.cat((embeddings, features), dim=0)

        return embeddings


class EcgDataAsTensor(nn.Module):
    # noinspection PyMethodMayBeStatic
    def forward(self, x: ECG) -> ECG:
        x.data = torch.from_numpy(x.data.get_data())
        return x


class EcgSequenceResampling(nn.Module):
    def __init__(self, original_fs: int, sequence_duration_seconds: int,
                 resampler: nn.Module = IDENTITY, channels_first: bool = False):
        super().__init__()
        self.sequence_length = original_fs * sequence_duration_seconds
        self.resampler: nn.Module = resampler
        self.channels_first = channels_first

    @timed()
    def forward(self, x: ECG) -> ECG:
        if self.channels_first:
            x.data = x.data.T

        segments = int(x.data.shape[0] / self.sequence_length)
        if x.data.shape[0] % self.sequence_length != 0:
            segments += 1

        y: Optional[torch.Tensor] = None
        for i in range(segments):
            x_i = x.data[i * self.sequence_length:(i + 1) * self.sequence_length]
            res = self.resampler(x_i)
            if self.channels_first:
                res = res.T
            # TODO Sablgiata questa concat (dovrebbe fare su un livello piu atlo
            res = res.unsqueeze(0)
            # We have new dimension that records the sequence.
            y: torch.Tensor = torch.cat((y, res)) if y is not None else res

        x.data = y
        return x

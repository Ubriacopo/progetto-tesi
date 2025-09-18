from __future__ import annotations

import dataclasses
from typing import List, Optional

import requests
import torch
from jedi.inference.gradual.typing import Callable
from torch import nn

from common.data.ecg.ecg import ECG


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
        ecg.data = ecg.data.tolist()
        return ECGPayload(
            signal=ecg.data,
            fs=ecg.fs,
            patient_age=ecg.patient_age,
            patient_gender=ecg.patient_gender,
            lead_names=ecg.lead_names
        )


class EcgFmEmbedderTransform(nn.Module):
    def __init__(self, data_transform_fn: Callable[[ECG], ECG], endpoint: str = "localhost:8000/extract_features"):
        super(EcgFmEmbedderTransform, self).__init__()
        self.endpoint: str = endpoint
        self.data_transform_fn = data_transform_fn

    def forward(self, x: ECG):
        r = requests.post(f"http://{self.endpoint}", json=ECGPayload.from_ecg(x, self.data_transform_fn))
        features = r.json()["features"]["values"]
        features = torch.tensor(features).float()
        return features

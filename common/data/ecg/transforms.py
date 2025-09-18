from __future__ import annotations
import dataclasses
from typing import List, Optional

import requests
from torch import nn

from common.data.ecg.ecg import ECG


@dataclasses.dataclass
class ECGPayload:
    signal: List[List[float]]  # ECG signal data: [leads, samples], e.g., [12, 5000]
    fs: Optional[int] = None  # Sampling rate in Hz (default: 500)
    # TODO: posso estrarli
    patient_age: Optional[int] = None  # Patient age in years
    patient_gender: Optional[str] = None  # Patient gender (M/F)"
    # TODO capire come prenderli
    lead_names: Optional[List[str]] = None  # Lead names (default: 12-lead standard)

    @staticmethod
    def from_ecg(ecg: ECG) -> ECGPayload:
        return ECGPayload(signal=ecg.data, fs=ecg.fs)


class EcgFmEmbedderTransform(nn.Module):
    def __init__(self, endpoint: str = "localhost:8000/extract_features"):
        super(EcgFmEmbedderTransform, self).__init__()
        self.endpoint: str = endpoint

    def forward(self, x: ECG):
        # todo extract what needed
        lead_names = ["I"]
        req_body = {"signal": x, "fs": 500, "lead_names": lead_names}
        r = requests.post(f"http://{self.endpoint}", json=req_body)

        return r.json()

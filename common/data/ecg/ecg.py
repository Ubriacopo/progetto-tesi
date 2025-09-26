import dataclasses
from typing import Optional

from common.data.signal.signal import Signal


@dataclasses.dataclass
class ECG(Signal):
    leads: list
    # Used is some instances of ECG analysis
    patient_age: Optional[int] = None  # Patient age in years
    patient_gender: Optional[str] = None  # Patient gender (M/F)"

    def export(self, base_path: str, output_path_to_relative: str = None):
        pass

    @staticmethod
    def modality_code() -> str:
        return "ecg"

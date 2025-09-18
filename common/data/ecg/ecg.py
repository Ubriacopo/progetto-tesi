# TODO: https://github.com/bowang-lab/ECG-FM/tree/main/data Va bene per missing LEAD da studio TolerantECG. Andrei per questo che è meno recente
# MA QUESTO MEGLIO PER MISSING LEAD https://github.com/Fsoft-AIC/TolerantECG (problema è molto recente)
# ELSE THIS ALSO https://github.com/PKUDigitalHealth/ECGFounder (Good base)
# Map your leads: RA & LA = effectively lead I. If you have RA & LL, you’d also get lead II.
from common.data.signal.signal import Signal


class ECG(Signal):
    leads: list

    def export(self, base_path: str, output_path_to_relative: str = None):
        pass

    @staticmethod
    def modality_code() -> str:
        return "ecg"

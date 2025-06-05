from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import loadmat
from torch import Tensor

from data.media.media import Media


class SignalDataCollector:
    # Contains all the Signal data loading steps
    def __init__(self):
        self.data, self.processed_data = pd.DataFrame(), pd.DataFrame()

    def load_resource(self, file_path: str, participant_id: str):
        resource = np.load(file_path, allow_pickle=True)
        df = pd.DataFrame({k: resource[k].unsqueeze() for k in resource.files})
        df["participant_id"] = participant_id
        self.data = pd.concat([self.data, df], ignore_index=True)


# The signal works on a file but contains more results.


class Signal(Media):
    """
        I won't for sure be able to read the full dataframe immediately.
        What I can do is load the entire file. Should I?
    """

    def get_value(self, raw: bool = False):
        # TODO Fast retrieval
        #       Probabilmente cosa piu facile sarebbe quella che Signal si tiene il suo valore.
        df = self.collector.data
        return self.collector.processed_data[df["participant_id"] == self.pid]

    # TODO Vedi come costruire questo
    def __init__(self, collector: SignalDataCollector, pid: str, file_path: str, trial: int, lazy: bool = True):
        self.collector = collector
        self.pid = pid  # Participant ID
        self.trial = trial
        super().__init__(file_path, lazy)

    def get_info(self):
        return {"file_path": self.file_path, "frequency": ""}

    def _inner_load(self, **kwargs):
        df = self.collector.data
        if df[df["participant_id"] == self.pid].empty:
            self.collector.load_resource(self.file_path, self.pid)

    def _inner_process(self, **kwargs):
        df = self.collector.processed_data
        if df[df["participant_id"] == self.pid].empty:
            pass  # TODO pre-process here.


def extract_trial_data(destination_path: str, source_path: str):
    """
    The data is converted to a numpy friendly type to help us work better (we have some advantages).
    Consideration:
        Load all EEG (40 × 100MB = ~4GB) into RAM at startup — easy with 128GB. (Our server)

    :param destination_path:
    :param source_path:
    """
    mat = loadmat(source_path)  # Source file

    data = {k: v for k, v in mat.items() if not k.startswith("__")}
    for key in data:
        # Remove the heading dimension
        data[key] = data[key].squeeze()

    file_name = Path(source_path).stem

    output_path = Path(f"{destination_path}/{file_name}.npz")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **data)

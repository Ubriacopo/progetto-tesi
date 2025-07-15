from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torchaudio
from torch.utils.data import Dataset

from common.amigos.sampler import StoredSamplingResult
from common.data.video import extract_frames
from common.ds.transform import Compose


# https://www.youtube.com/watch?v=LevmcKxZAmY
class AMIGOSDataset(Dataset):
    def __init__(self, dataset_spec_file: str, video_transform: Compose = None, audio_transform: Compose = None,
                 text_transform: Compose = None, eeg_transform: Compose = None):
        # Transforms for the data.
        self.video_transform: Compose = video_transform
        self.audio_transform: Compose = audio_transform
        self.text_transform: Compose = text_transform
        self.eeg_transform: Compose = eeg_transform

        self.descriptor = pd.read_csv(dataset_spec_file)
        files = self.descriptor["data_file"].unique().tolist()
        # We can keep the eeg data cached? todo
        self.cached_eeg = np.array([np.load(f) for f in files])

    def __getitem__(self, index: int):
        record = self.descriptor.iloc[index].to_dict()
        entry = StoredSamplingResult(**record)

        v = extract_frames(cv2.VideoCapture(entry.media_path))
        if self.video_transform is not None:
            v = self.video_transform(v)

        waveform, sample_rate = torchaudio.load("audio.mp3")
        if self.audio_transform is not None:
            waveform, sample_rate = self.audio_transform([waveform, sample_rate])

        # todo che fare con il testo?
        # Devo fare transcript della clip
        t = ""
        if self.text_transform is not None:
            t = self.text_transform(waveform)

        eeg = self.cached_eeg[int(Path(entry.data_file).stem.split("_")[-1])][entry.data_index]
        if self.eeg_transform is not None:
            eeg = self.eeg_transform(waveform)

        # Unsupervised so we return none as y
        return v, waveform, t, eeg, None

    def __len__(self):
        return len(self.descriptor)


# todo transforms per i dati? Passabili
class PersistableAMIOGSDataset(AMIGOSDataset):
    pass
    # Questo permette il salvataggio in embeddings

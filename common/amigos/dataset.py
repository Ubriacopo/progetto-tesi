from abc import ABC
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torchaudio
from torch.utils.data import Dataset

from common.amigos.sampler import StoredSamplingResult
from common.data.extensions import video_extensions, audio_extensions, text_extensions
from common.data.video import extract_frames
from common.ds.transform import Compose

# todo make abstract class to recycle the extension tracking
class AMIGOSDataset(Dataset, ABC):
    def __init__(self, dataset_spec_file: str, video_transform: Compose = None,
                 audio_transform: Compose = None, text_transform: Compose = None,
                 eeg_transform: Compose = None, persist_while_fetching: bool = False):
        super().__init__()
        # If true when calling the transform I store a copy somewhere. TODO Choose policy
        self.persist_while_fetching: bool = persist_while_fetching

        self.video_transform: Compose = video_transform
        self.audio_transform: Compose = audio_transform
        self.text_transform: Compose = text_transform
        self.eeg_transform: Compose = eeg_transform

        self.descriptor = pd.read_csv(dataset_spec_file)

        files = self.descriptor["data_file"].unique().tolist()
        # We can keep the eeg data cached? todo Might be much.
        # TODO Enhancement: Rotational cached eeg data.
        #  I keep only half in memory and read on demand discarding the oldest or something like that.
        self.cached_eeg = np.array([np.load(f) for f in files])

        # Only during init I have to check if samples have txt and audio to see what to serve
        media_path = self.descriptor.iloc[0].to_dict()["media_path"]
        path = Path(media_path)
        extensions = set([p.suffix for p in list(path.parent.glob("0.*"))])

        # One of these can be missing but not the EEG data of course.
        text_ext = extensions & set(text_extensions())
        if len(text_ext) > 1:
            raise AssertionError("We only support one text extension at a time")
        self.text_enabled = bool(text_ext)

        video_ext = extensions & set(video_extensions())
        if len(text_ext) > 1:
            raise AssertionError("We only support one video extension at a time")
        self.video_enabled = bool(video_ext)

        audio_ext = extensions & set(audio_extensions())
        if len(text_ext) > 1:
            raise AssertionError("We only support one audio extension at a time")
        self.audio_enabled = bool(audio_ext)

        self.extensions = dict(video=video_ext, audio=audio_ext, text=text_ext)

    def __getitem__(self, index: int):
        record = self.descriptor.iloc[index].to_dict()
        entry = StoredSamplingResult(**record)

        video_data = None
        if self.video_enabled:
            (video_ext,) = self.extensions["video"]
            video_data = extract_frames(cv2.VideoCapture(entry.data_file + video_ext))
            if self.video_transform is not None:
                video_data = self.video_transform(video_data)

        waveform, sample_rate = None, None
        if self.audio_enabled:
            (audio_ext,) = self.extensions["audio"]
            waveform, sample_rate = torchaudio.load(entry.data_file + audio_ext)
            if self.audio_transform is not None:
                waveform, sample_rate = self.audio_transform([waveform, sample_rate])

        text: str | None = None
        if self.text_enabled:
            (text_ext,) = self.extensions["text"]
            with open(entry.data_file + text_ext) as f:
                text = f.read()
            if self.text_transform is not None:
                text = self.text_transform(text)

        eeg = self.cached_eeg[int(Path(entry.data_file).stem.split("_")[-1])][entry.data_index]
        if self.eeg_transform is not None:
            eeg = self.eeg_transform(eeg)
        # TODO: Vedi se ritornare None in dataset va bnee
        return video_data, waveform, text, eeg, None

    def __len__(self):
        return len(self.descriptor)
    # https://www.youtube.com/watch?v=LevmcKxZAmY


# todo transforms per i dati? Passabili
class PersistableAMIOGSDataset(AMIGOSDataset):
    pass
    # Questo permette il salvataggio in embeddings

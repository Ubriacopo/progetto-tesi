import dataclasses
from abc import ABC
from typing import Optional

import mne.io
import pandas as pd
import torch
import torchaudio
from torch import device

from common.data.audio import Audio
from common.data.audio.transforms import AudioToTensor
from common.data.data_point import EEGDatasetDataPoint
from common.data.eeg import EEG
from common.data.eeg.mne_utils import find_segment_by_descriptor
from common.data.eeg.transforms import EEGToTensor
from common.data.text import Text
from common.data.transform import KwargsCompose
from common.data.video import Video, VideoToTensor


class EEGPdSpecMediaDataset(torch.utils.data.Dataset, ABC):
    def __init__(self,
                 dataset_spec_file: str,
                 eeg_transform: KwargsCompose,
                 selected_device: device = None,
                 video_transform: KwargsCompose = None,
                 audio_transform: KwargsCompose = None,
                 text_transform: KwargsCompose = None):
        super().__init__()
        # Auto device selection.
        if selected_device is None:
            selected_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device: device = selected_device

        df = pd.read_csv(dataset_spec_file, index_col=False)
        self.objects = [EEGDatasetDataPoint.from_dict(d) for d in df.to_dict(orient="records")]

        if eeg_transform is None:
            raise ValueError("EEG transform must be defined")
        self.eeg_transform: Optional[KwargsCompose] = eeg_transform
        self.vid_transform: Optional[KwargsCompose] = video_transform
        self.aud_transform: Optional[KwargsCompose] = audio_transform
        self.txt_transform: Optional[KwargsCompose] = text_transform

    def load_vid(self, x: Video, idx: int) -> tuple[torch.Tensor, dict] | None:
        if x is None:
            return None

        x, metadata = VideoToTensor()(x)
        x, metadata = self.vid_transform(x, **metadata)

        return x, metadata

    def load_aud(self, x: Audio, idx: int) -> tuple[torch.Tensor, dict] | None:
        if x is None:
            return None

        x, metadata = AudioToTensor()(x)
        x, metadata = self.aud_transform(x, **metadata)

        return x, metadata

    def load_txt(self, x: Text, idx: int) -> tuple[torch.Tensor, dict] | None:
        if x is None: return None

        with open(x.file_path) as f:
            x.data = f.read()

        metadata = x.to_dict(metadata_only=True)
        x, metadata = self.txt_transform(x, **metadata)

        return x, metadata

    def load_eeg(self, x: EEG, idx: int, eid: str) -> tuple[torch.Tensor, dict] | None:
        if x is None:
            raise RuntimeError("EEG data is None but that cannot happen.")

        x, metadata = EEGToTensor()(x, entry_id=eid)
        x, metadata = self.eeg_transform(x, **metadata)

        return x

    def __getitem__(self, idx: int) -> EEGDatasetDataPoint:
        template = self.objects[idx]
        # For the moment to avoid caching the stuff in data we do this workaround.
        # We will se if there is some sort of caching required somewhere.
        x = EEGDatasetDataPoint(
            entry_id=template.entry_id,
            vid=dataclasses.replace(template.vid, data=None) if template.vid is not None else None,
            aud=dataclasses.replace(template.aud, data=None) if template.aud is not None else None,
            txt=dataclasses.replace(template.txt, data=None) if template.txt is not None else None,
            eeg=dataclasses.replace(template.eeg, data=None) if template.eeg is not None else None,
        )

        x.vid = self.load_vid(x.vid, idx)
        x.aud = self.load_aud(x.aud, idx)
        x.txt = self.load_txt(x.txt, idx)
        x.eeg = self.load_eeg(x.eeg, idx, x.entry_id)

        return x

    def __len__(self):
        return len(self.objects)

import dataclasses
from abc import ABC, abstractmethod
from typing import Optional, Tuple

import pandas as pd
import torch
from torch import device, nn

from common.data.audio import Audio
from common.data.audio.transforms import AudioToTensor
from common.data.data_point import EEGDatasetDataPoint, EEGModalityComposeWrapper, call_pipelines
from common.data.eeg import EEG
from common.data.eeg.transforms import EEGToTensor
from common.data.text import Text
from common.data.video import Video, VideoToTensor


class EEGMediaDataset(torch.utils.data.Dataset, ABC):
    def load_vid(self, x: Video, idx: int) -> tuple[torch.Tensor, dict] | None:
        return VideoToTensor()(x) if x is not None else None

    def load_aud(self, x: Audio, idx: int) -> tuple[torch.Tensor, dict] | None:
        return AudioToTensor()(x) if x is not None else None

    def load_txt(self, x: Text, idx: int) -> tuple[torch.Tensor, dict] | None:
        # todo quando usiamo testo pensare
        if x is None: return None

        with open(x.file_path) as f:
            x.data = f.read()

        return x

    def load_eeg(self, x: EEG, idx: int) -> tuple[torch.Tensor, dict] | None:
        if x is None:
            raise RuntimeError("EEG data is None but that cannot happen.")
        return EEGToTensor()(x)

    @abstractmethod
    def __getitem__(self, idx: int) -> EEGDatasetDataPoint:
        pass


class EEGPdSpecMediaDataset(EEGMediaDataset, ABC):
    def __init__(self, dataset_spec_file: str, transforms: EEGModalityComposeWrapper, selected_device: device = None):
        super().__init__()
        # Auto device selection.
        if selected_device is None:
            selected_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.device: device = selected_device
        df = pd.read_csv(dataset_spec_file, index_col=False)
        self.objects = [EEGDatasetDataPoint.from_dict(d) for d in df.to_dict(orient="records")]

        if transforms.eeg_transform is None:
            raise ValueError("EEG transform must be defined")

        self.base_transforms: EEGModalityComposeWrapper = transforms

        self.eeg_transform: Optional[nn.Sequential] = transforms.eeg_transform
        self.vid_transform: Optional[nn.Sequential] = transforms.vid_transform
        self.aud_transform: Optional[nn.Sequential] = transforms.aud_transform
        self.txt_transform: Optional[nn.Sequential] = transforms.txt_transform

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
        x.eeg = self.load_eeg(x.eeg, idx)
        x = call_pipelines(x, self.base_transforms)

        return x

    def __len__(self):
        return len(self.objects)


class KDEEGPdSpecMediaDataset(EEGPdSpecMediaDataset, ABC):
    def __init__(self, dataset_spec_file: str,
                 shared_transform: EEGModalityComposeWrapper,
                 modality_transforms: list[EEGModalityComposeWrapper]):
        super().__init__(dataset_spec_file, shared_transform)
        if modality_transforms is None or len(modality_transforms) < 2:
            raise ValueError("modality_transforms must have at least 2 elements you swine!/j"
                             "Use the KD less one if you have one modality output at a time.")
        self.multi_out_transforms = modality_transforms

    def __getitem__(self, idx: int) -> Tuple:
        x = super().__getitem__(idx)

        outputs = []
        for mod in range(len(self.multi_out_transforms)):
            # Replace where possible
            y = dataclasses.replace(x)
            outputs.append(call_pipelines(y, self.multi_out_transforms[mod]))

        return tuple(outputs)

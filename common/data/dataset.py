import dataclasses
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd
import torch
from torch import device, nn

from common.data.data_point import EEGDatasetDataPoint, EEGDatasetTransformWrapper, call_pipelines


class EEGMediaDataset(torch.utils.data.Dataset, ABC):
    @abstractmethod
    def __getitem__(self, idx: int) -> EEGDatasetDataPoint:
        pass

# TODO Agnostic one too
class EEGPdSpecMediaDataset(EEGMediaDataset, ABC):
    def __init__(self, dataset_spec_file: str, transforms: EEGDatasetTransformWrapper, selected_device: device = None):
        super().__init__()
        # Auto device selection.
        if selected_device is None:
            selected_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.device: device = selected_device
        df = pd.read_csv(dataset_spec_file, index_col=False)

        self.base_path: str = str(Path(dataset_spec_file).parent)
        self.objects = [EEGDatasetDataPoint.from_dict(d, self.base_path) for d in df.to_dict(orient="records")]

        if transforms.eeg_transform is None:
            raise ValueError("EEG transform must be defined")

        self.base_transforms: EEGDatasetTransformWrapper = transforms

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

        x = call_pipelines(x, self.base_transforms)
        return x

    def __len__(self):
        return len(self.objects)


def fix(x):  # replace None with empty tensor
    return x if x is not None else torch.empty(0)


class KDEEGPdSpecMediaDataset(EEGPdSpecMediaDataset):
    def __init__(self, dataset_spec_file: str,
                 shared_transform: EEGDatasetTransformWrapper,
                 modality_transforms: list[EEGDatasetTransformWrapper]):
        super().__init__(dataset_spec_file, shared_transform)
        if modality_transforms is None or len(modality_transforms) < 2:
            raise ValueError("modality_transforms must have at least 2 elements you swine!/j"
                             "Use the KD less one if you have one modality output at a time.")
        self.multi_out_transforms = modality_transforms

    def __getitem__(self, idx: int) -> dict:
        x = super().__getitem__(idx)
        outputs = {}
        for mod in range(len(self.multi_out_transforms)):
            # Replace where possible
            y = dataclasses.replace(x)
            o = call_pipelines(y, self.multi_out_transforms[mod])

            out = {
                "eeg": fix(o.eeg), "vid": fix(o.vid), "aud": fix(o.aud), "txt": fix(o.txt),
                "mask": torch.Tensor([o.eeg is not None, o.vid is not None, o.aud is not None, o.txt is not None, ]),
                "order": ("eeg", "vid", "aud", "txt", "mask"),
            }
            outputs[self.multi_out_transforms[mod].name] = out

        return outputs

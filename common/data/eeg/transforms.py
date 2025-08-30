import dataclasses
from typing import Optional

import mne
import torch

from common.data.data_point import EEGDatasetDataPoint
from .eeg import EEG
from .mne_utils import find_segment_by_descriptor


@dataclasses.dataclass
class EEGToTensor:
    take_eeg: bool = True
    take_ecg: bool = False

    def __call__(self, x: EEG, entry_id: str, device: str = "cpu", *args, **kwargs) -> tuple[torch.Tensor, dict]:
        raw: Optional[mne.io.BaseRaw] = x.data
        if raw is None:
            fif = mne.io.read_raw_fif(x.file_path)
            segments = find_segment_by_descriptor(fif, entry_id)
            if len(segments) == 0 or len(segments) > 1:
                raise RuntimeError(f"Found {len(segments)} EEG for {entry_id}. There an error with {entry_id}.")
            _, onset, duration = segments[0]
            raw = fif.copy().crop(tmin=onset, tmax=onset + duration)

        metadata = x.to_dict(metadata_only=True)

        picks = mne.pick_types(raw.info, eeg=self.take_eeg, ecg=self.take_ecg)
        x = torch.from_numpy(raw.get_data(picks))

        return x, kwargs | metadata


# todo meh rewrite for EEG only avoid using EEGDatasetDataPoint
@dataclasses.dataclass
class EEGToMneRawFromChannels:
    channel_names: list[str]
    channel_types: list[str]

    def __call__(self, x: EEGDatasetDataPoint | EEG, verbose: bool = False):
        e: EEG = x.eeg if isinstance(x, EEGDatasetDataPoint) else x
        if e.data is None:
            raise ValueError("EEG data has to be loaded to transform to mne")

        info = mne.create_info(ch_names=self.channel_names, ch_types=self.channel_types, sfreq=e.fs)
        raw = mne.io.RawArray(e.data, info=info, verbose=verbose)

        e.data = raw
        return x


class EEGMneAddAnnotation:
    def __call__(self, x: EEGDatasetDataPoint | EEG, description: str = None):
        e: EEG = x.eeg if isinstance(x, EEGDatasetDataPoint) else x

        raw: mne.io.BaseRaw = e.data

        if not isinstance(raw, mne.io.BaseRaw):
            raise TypeError("To call this pipeline you have to turn to MNE object first ")

        description = x.entry_id if isinstance(x, EEGDatasetDataPoint) else description
        if description is None:
            raise TypeError("A valid descriptor to identify the annotation is required")

        start, stop = e.interval
        new_annotation = mne.Annotations(onset=[start], duration=[stop - start], description=x.entry_id, orig_time=None)
        existing = getattr(raw, 'annotations', None)

        merged = new_annotation if existing is None else existing + new_annotation
        raw.set_annotations(merged)
        return x

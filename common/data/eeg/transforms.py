from typing import Optional

import mne
import torch

from common.data.data_point import EEGDatasetDataPoint
from .eeg import EEG
from .mne_utils import find_segment_by_descriptor
from ..transform import CustomBaseTransform


class EEGToTensor(CustomBaseTransform):
    def __init__(self, take_eeg: bool = True, take_ecg: bool = False):
        super().__init__()
        self.take_eeg: bool = take_eeg
        self.take_ecg: bool = take_ecg

    def do(self, x: EEG) -> tuple[torch.Tensor, dict]:
        raw: Optional[mne.io.BaseRaw] = x.data
        if raw is None:
            fif = mne.io.read_raw_fif(x.file_path)
            segments = find_segment_by_descriptor(fif, x.entry_id)
            if len(segments) == 0 or len(segments) > 1:
                raise RuntimeError(f"Found {len(segments)} EEG for {x.entry_id}. There an error with {x.entry_id}.")
            _, onset, duration = segments[0]
            raw = fif.copy().crop(tmin=onset, tmax=onset + duration)

        metadata = x.to_dict(metadata_only=True)

        picks = mne.pick_types(raw.info, eeg=self.take_eeg, ecg=self.take_ecg)
        x = torch.from_numpy(raw.get_data(picks))

        return x, metadata


class EEGToMneRawFromChannels(CustomBaseTransform):
    def __init__(self, channel_names: list[str], channel_types: list[str], verbose: bool = False):
        super().__init__()
        self.channel_names: list[str] = channel_names
        self.channel_types: list[str] = channel_types
        self.verbose: bool = verbose

    def do(self, x: EEG):
        if x.data is None:
            raise ValueError("EEG data has to be loaded to transform to mne")

        info = mne.create_info(ch_names=self.channel_names, ch_types=self.channel_types, sfreq=x.fs)
        raw = mne.io.RawArray(x.data, info=info, verbose=self.verbose)
        x.data = raw
        return x


class EEGMneAddAnnotation(CustomBaseTransform):
    @classmethod
    def scriptable(cls) -> bool:
        return False

    def do(self, x: EEG):
        raw: mne.io.BaseRaw = x.data
        if not isinstance(raw, mne.io.BaseRaw):
            raise TypeError("To call this pipeline you have to turn to MNE object first ")

        if x.entry_id is None:
            raise TypeError("A valid descriptor to identify the annotation is required")

        start, stop = x.interval
        new_annotation = mne.Annotations(onset=[start], duration=[stop - start], description=x.entry_id, orig_time=None)
        existing = getattr(raw, 'annotations', None)

        merged = new_annotation if existing is None else existing + new_annotation
        raw.set_annotations(merged)
        return x

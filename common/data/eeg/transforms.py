import dataclasses

import mne

from .eeg import EEG
from common.data.data_point import EEGDatasetDataPoint


@dataclasses.dataclass
class EEGToMneRawFromChannels:
    channel_names: list[str]
    channel_types: list[str]

    def __call__(self, x: EEGDatasetDataPoint | EEG, verbose: bool = False):
        e: EEG = x.eeg if isinstance(x, EEGDatasetDataPoint) else x
        assert e.data is not None, "Data has to be loaded"

        info = mne.create_info(ch_names=self.channel_names, ch_types=self.channel_types, sfreq=e.fs)
        raw = mne.io.RawArray(e.data, info=info, verbose=verbose)

        e.data = raw
        return x


class EEGMneAddAnnotation:
    def __call__(self, x: EEGDatasetDataPoint | EEG, description: str = None):
        e: EEG = x.eeg if isinstance(x, EEGDatasetDataPoint) else x

        raw: mne.io.BaseRaw = e.data
        assert isinstance(raw, mne.io.BaseRaw), "To call this pipeline you have to turn to MNE object first"
        description = x.entry_id if isinstance(x, EEGDatasetDataPoint) else description
        assert description is not None, "A valid descriptor to identify the annotation is required"

        start, stop = e.interval
        new_annotation = mne.Annotations(onset=[start], duration=[stop - start], description=x.entry_id, orig_time=None)
        existing = getattr(raw, 'annotations', None)

        merged = new_annotation if existing is None else existing + new_annotation
        raw.set_annotations(merged)
        return x

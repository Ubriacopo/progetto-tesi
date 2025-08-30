import dataclasses
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

import cv2
import mne.io
import numpy as np
import pandas as pd
import torch
import torchaudio
from torch import device

from common.data.audio import Audio
from common.data.data_point import EEGDatasetDataPoint
from common.data.eeg import EEG
from common.data.eeg.mne_utils import find_segment_by_descriptor
from common.data.extensions import text_extensions, video_extensions, audio_extensions
from common.data.sampler import SamplingDescriptor
from common.data.text import Text
from common.data.transform import Compose
from common.data.video import Video
from common.data.video.utils import extract_frames


# todo ds usando sample_container

@dataclasses.dataclass
class DatasetRecord:
    eeg: dict | torch.Tensor | None
    video: dict | torch.Tensor | None
    audio: dict | torch.Tensor | None
    # Texts to be tokenized with the relative attention mask.
    text: dict | torch.Tensor | None
    present: torch.Tensor  # Always defined.


def check_extension(media_path: Path, subset: set[str], multi: bool = False) -> tuple[bool, set[str]]:
    extensions = set([p.suffix for p in list(media_path.parent.glob("0.*"))])
    matches = extensions & subset

    if not multi and len(matches) > 1:
        raise AssertionError("We only support one extension at a time")

    return bool(matches), matches


def get_filepath(media_path: str, extensions: set[str]) -> str | None:
    for extensions in extensions:
        if Path(media_path + extensions).is_file():
            return str(media_path + extensions)
    return None


class SimpleMediaBasedDataset(torch.utils.data.Dataset, ABC):
    """
    Media based data. Medias are processed from files (could be overlapping by overriding the logic to get the data).
    For now, it simply loads video as frames, audio as wave + sr, transcript as text and eeg data by end defined logic.
    """

    def __init__(self, train: bool = True, video_transform: Compose = None,
                 audio_transform: Compose = None, text_transform: Compose = None, eeg_transform: Compose = None):
        super().__init__()

        # We work with these kinds of data.
        self.vid_transform: Optional[Compose] = video_transform
        self.aud_transform: Optional[Compose] = audio_transform
        self.txt_transform: Optional[Compose] = text_transform
        if eeg_transform is None:
            raise ValueError("EEG transform must be defined")

        self.eeg_transform: Optional[Compose] = eeg_transform

        self.train: bool = train

        # Contains the sets of extensions to check
        self.extensions = dict()
        media_path = Path(self.get_media_path(0))
        # If a transform passed is None the modality is considered disabled.
        self.txt_enabled, self.extensions["text"] = check_extension(media_path, text_extensions())
        self.txt_enabled = self.txt_enabled and self.txt_transform is not None

        self.vid_enabled, self.extensions["video"] = check_extension(media_path, video_extensions())
        self.vid_enabled = self.vid_enabled and self.vid_transform is not None

        self.aud_enabled, self.extensions["audio"] = check_extension(media_path, audio_extensions())
        self.aud_enabled = self.aud_enabled and self.aud_transform is not None

    @abstractmethod
    def get_media_path(self, idx: int) -> str:
        """
        Returns path of the media file.

        :param idx: Index of the media file
        :return: Path of the media file
        """
        pass

    @abstractmethod
    def get_eeg(self, idx: int) -> Optional[torch.Tensor]:
        pass

    def get_video(self, idx: int) -> list[torch.Tensor] | None:
        if not self.vid_enabled:
            return None

        media_path = self.get_media_path(idx)
        path = get_filepath(media_path, self.extensions["video"])

        vd = extract_frames(cv2.VideoCapture(path))
        vd = self.vid_transform(vd, train=self.train, return_both=False)

        return vd

    def get_audio(self, idx: int) -> Optional[tuple[torch.Tensor, int]]:
        if not self.aud_enabled:
            return None

        media_path = self.get_media_path(idx)
        path = get_filepath(media_path, self.extensions["audio"])

        ad = torchaudio.load(path)
        ad = self.aud_transform(ad[0], train=self.train)

        return ad

    def get_text(self, idx: int) -> None | str:
        if not self.txt_enabled:
            return None

        media_path = self.get_media_path(idx)
        path = get_filepath(media_path, self.extensions["text"])

        with open(path) as f:
            td = f.read()

        td = self.txt_transform(td, train=self.train)

        return td

    def __getitem__(self, idx: int) -> DatasetRecord:
        # If we use cache and element has been cached previously.
        vd = self.get_video(idx, )
        ad = self.get_audio(idx, )
        td = self.get_text(idx, )
        ed = self.get_eeg(idx, )

        present = [ed is not None, vd is not None, ad is not None, td is not None]
        return DatasetRecord(eeg=ed, video=vd, audio=ad, text=td, present=torch.tensor(present, dtype=torch.bool))


class SpecMediaBasedDataset(SimpleMediaBasedDataset, ABC):

    def __init__(self, dataset_spec_file: str,
                 train: bool = True, video_transform: Compose = None,
                 audio_transform: Compose = None, text_transform: Compose = None, eeg_transform: Compose = None):
        """
        Media based dataset that uses a specification file (descriptor) (csv).
        It contains the condensed access info to all data relative to the dataset.

        :param dataset_spec_file:
        :param train:
        :param video_transform:
        :param audio_transform:
        :param text_transform:
        :param eeg_transform:
        """
        self.spec_file_path: str = dataset_spec_file

        self.spec = pd.read_csv(self.spec_file_path)
        files = self.spec["data_file"].unique().tolist()
        # TODO: Find a way to lazy load np.load
        # Read the EEG data
        self.cached_eeg = [np.load(f, allow_pickle=True) for f in files]
        super().__init__(train, video_transform, audio_transform, text_transform, eeg_transform)

    def register_in_cache(self, idx: int) -> None:
        self.spec["cached"].iloc[idx] = True
        self.spec.to_csv(self.spec_file_path, index=False)

    def get_eeg(self, idx: int, cached: bool = False) -> None | torch.Tensor:
        entry = self.get_entry(idx)

        ed = self.cached_eeg[int(Path(entry.data_file).stem.split("_")[-1])][entry.data_index]
        ed = self.eeg_transform(ed, train=self.train)

        return ed

    def get_entry(self, idx: int):
        return SamplingDescriptor(**(self.spec.iloc[idx].to_dict()))

    def get_media_path(self, idx: int) -> str:
        return self.get_entry(idx).media_path


class EEGPdSpecMediaDataset(torch.utils.data.Dataset, ABC):
    def __init__(self, dataset_spec_file: str, eeg_transform: Compose, selected_device: device = None,
                 video_transform: Compose = None, audio_transform: Compose = None, text_transform: Compose = None):
        super().__init__()
        # Auto device selection.
        if selected_device is None:
            selected_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device: device = selected_device

        df = pd.read_csv(dataset_spec_file, index_col=False)
        self.objects = [EEGDatasetDataPoint.from_dict(d) for d in df.to_dict(orient="records")]

        if eeg_transform is None:
            raise ValueError("EEG transform must be defined")
        self.eeg_transform: Optional[Compose] = eeg_transform
        self.vid_transform: Optional[Compose] = video_transform
        self.aud_transform: Optional[Compose] = audio_transform
        self.txt_transform: Optional[Compose] = text_transform

    def load_vid(self, x: Video, idx: int) -> Optional[Video]:
        if x is None:
            return None

        # VideoDecoder(x.file_path, device=self.device) doesn't work
        x.data = extract_frames(cv2.VideoCapture(x.file_path))
        x.data = self.vid_transform(x)

        return x

    def load_aud(self, x: Audio, idx: int) -> Optional[Audio]:
        if x is None:
            return None

        x.data = torchaudio.load(x.file_path)
        x.data = self.aud_transform(x)

        return x

    def load_txt(self, x: Text, idx: int) -> Optional[Text]:
        if x is None:
            return None

        with open(x.file_path) as f:
            x.data = f.read()

        x.data = self.txt_transform(x)
        return x

    def load_eeg(self, x: EEG, idx: int, eid: str) -> Optional[EEG]:
        if x is None:
            raise RuntimeError("EEG data is None but that cannot happen.")

        fif = mne.io.read_raw_fif(x.file_path)
        segments = find_segment_by_descriptor(fif, eid)
        if len(segments) == 0 or len(segments) > 1:
            raise RuntimeError(f"Found {len(segments)} EEG for {eid}. There is something wrong with {eid}.")

        _, onset, duration = segments[0]
        x.data = fif.copy().crop(tmin=onset, tmax=onset + duration)
        x.data = self.eeg_transform(x)

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

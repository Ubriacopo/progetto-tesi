import dataclasses
from abc import ABC, abstractmethod
from pathlib import Path

import cv2
import numpy as np
import torch
import torchaudio
from safetensors import safe_open
from safetensors.torch import save_file

from common.data.extensions import text_extensions, video_extensions, audio_extensions
from common.data.video import extract_frames
from common.ds.transform import Compose


@dataclasses.dataclass
class TextEntry:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor


@dataclasses.dataclass
class DatasetRecord:
    eeg: None | torch.Tensor
    video: None | torch.Tensor
    audio: None | torch.Tensor
    # Texts to be tokenized with the relative attention mask.
    text: None | TextEntry | torch.Tensor
    present: None | torch.Tensor


class SimpleLoaderDataset(torch.utils.data.Dataset):
    def __init__(self, folds: list[list]):
        self.folds = folds

    def __getitem__(self, item: int):
        # Se manca un testo posso ricevere [num, num, None, num]. Multimodal safe?
        return tuple(lst[item] if len(lst) > item else None for lst in self.folds)

    def __len__(self):
        return len(self.folds[0])


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


class PersistingDataset(torch.utils.data.Dataset, ABC):
    """
    Summary here.
    """

    def __init__(self, use_cache: bool = True, video_transform: Compose = None, audio_transform: Compose = None,
                 text_transform: Compose = None, eeg_transform: Compose = None, train: bool = True):
        self.use_cache: bool = use_cache
        # We work with these kinds of data.
        self.video_transform: Compose = video_transform
        self.audio_transform: Compose = audio_transform
        self.text_transform: Compose = text_transform
        self.eeg_transform: Compose = eeg_transform

        self.train: bool = train

    @abstractmethod
    def persist_on_fetch(self, tensor: torch.Tensor, idx: int):
        pass

    @abstractmethod
    def get_entry(self, index: int):
        pass


class MediaBasedDataset(PersistingDataset, ABC):
    """
    Media based dataset. Medias are processed from files (could be overlapping by overriding the logic to get the data).
    For now, it simply loads video as frames, audio as wave + sr, transcript as text and eeg data by end defined logic.
    Being persisting once a record is called (in use_cache mode) it is persisted somewhere defined by the child class.
    All entries to be stored are assumed to be instanced as torch.Tensor
    """

    def __init__(self, use_cache: bool = True,
                 video_transform: Compose = None, audio_transform: Compose = None,
                 text_transform: Compose = None, eeg_transform: Compose = None, train: bool = True):
        super().__init__(use_cache, video_transform, audio_transform, text_transform, eeg_transform, train)
        # Contains the sets of extensions to check
        self.extensions = dict()
        media_path = Path(self.get_media(0))
        self.text_enabled, self.extensions["text"] = check_extension(media_path, text_extensions())
        self.video_enabled, self.extensions["video"] = check_extension(media_path, video_extensions())
        self.audio_enabled, self.extensions["audio"] = check_extension(media_path, audio_extensions())

    @abstractmethod
    def get_media(self, idx: int) -> str:
        pass

    def get_video(self, idx: int) -> list[np.ndarray] | None:
        """

        :param idx:
        :return: The list of frames of our video. The shape and channels are variable of course.
        """
        if not self.video_enabled:
            return None

        media_path = self.get_media(idx)
        path = get_filepath(media_path, self.extensions["video"])
        return extract_frames(cv2.VideoCapture(path))

    def get_audio(self, idx: int) -> None | tuple[torch.Tensor, int]:
        """

        :param idx:
        :return: Waveform and sampling rate tuple of the loaded audio.
        """
        if not self.audio_enabled:
            return None

        media_path = self.get_media(idx)
        path = get_filepath(media_path, self.extensions["audio"])
        return torchaudio.load(path)

    def get_text(self, idx: int) -> None | str:
        if not self.text_enabled or self.text_transform is None:
            return None

        media_path = self.get_media(idx)
        path = get_filepath(media_path, self.extensions["text"])
        with open(path) as f:
            return f.read()

    @abstractmethod
    def get_eeg(self, idx: int) -> None | torch.Tensor:
        pass

    def _restore_from_persistent(self, transform: Compose, cached: DatasetRecord, prop: str) -> DatasetRecord:
        if transform is not None and hasattr(cached, prop) and cached.__getattribute__(prop) is not None:
            res = transform.transform_but_skip_pre(cached.__getattribute__(prop), train=self.train, return_both=False)
            cached.__setattr__(prop, res)
        return cached

    def __getitem__(self, idx: int) -> DatasetRecord:
        cached = self.retrieve_from_persistent(idx)
        if cached is not None:
            cached = self._restore_from_persistent(self.video_transform, cached, prop="video")
            cached = self._restore_from_persistent(self.audio_transform, cached, prop="audio")
            cached = self._restore_from_persistent(self.text_transform, cached, prop="text")
            cached = self._restore_from_persistent(self.eeg_transform, cached, prop="audio")
            return cached

        cache_object = dict(video=None, audio=None, text=None, eeg=None)
        # Returns the loaded video in its format ready to pass to the transform
        vd = self.get_video(idx)
        if self.video_transform is not None and vd is not None:
            vd = self.video_transform(vd, train=self.train, return_both=self.use_cache)
            if self.use_cache:
                vd, cache_object["video"] = vd

        ad = self.get_audio(idx)  # Returns wavelength + freq
        if self.audio_transform is not None and ad is not None:
            ad = self.audio_transform(ad[0], train=self.train, return_both=self.use_cache)
            if self.use_cache:
                ad, cache_object["audio"] = ad

        td = self.get_text(idx)
        if self.text_transform is not None and td is not None:
            td = self.text_transform(td, train=self.train, return_both=self.use_cache)
            if self.use_cache:
                td, cache_object["text"] = td

        ed = self.get_eeg(idx)
        if self.eeg_transform is not None and ed is not None:
            ed = self.eeg_transform(ed, train=self.train, return_both=self.use_cache)
            if self.use_cache:
                ed, cache_object["eeg"] = ed

        present = [ed is not None, vd is not None, ad is not None, td is not None]
        record = DatasetRecord(eeg=ed, video=vd, audio=ad, text=td, present=torch.tensor(present, dtype=torch.bool))

        if self.use_cache:
            cache_object["present"] = record.present
            self.persist_on_fetch(DatasetRecord(**cache_object), self.get_persistent_path(idx))

        return record

    def persist_on_fetch(self, record: DatasetRecord, output: str):
        o = dict()
        for k, v in dataclasses.asdict(record).items():
            if v is not None:
                if isinstance(v, list):
                    # If we have a list
                    v = torch.stack(v)
                o[k] = v.detach().cpu()
        save_file(o, f"{output}.safetensors")

    def retrieve_from_persistent(self, idx: int) -> DatasetRecord | None:
        """
        Returns a dataset record by index if it finds. Else returns None.
        :param idx:
        :return:
        """
        path = self.get_persistent_path(idx) + ".safetensors"
        if not Path(path).exists() or not Path(path).is_file():
            return None

        with safe_open(path, framework="pt") as file:
            record = DatasetRecord(None, None, None, None, None)
            # Has to completely overlap
            for key in file.keys():
                record.__setattr__(key, file.get_tensor(key))
            return record

    @abstractmethod
    def get_persistent_path(self, index: int) -> str:
        pass

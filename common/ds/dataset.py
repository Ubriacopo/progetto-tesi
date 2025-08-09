import dataclasses
from abc import ABC, abstractmethod
from pathlib import Path

import cv2
import torch
import torchaudio

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
    text: None | TextEntry
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

    if not multi and len(matches):
        raise AssertionError("We only support one extension at a time")

    return bool(matches), matches


def get_filepath(media_path: str, extensions: set[str]) -> str | None:
    for extensions in extensions:
        if Path(media_path + extensions).is_file():
            return str(media_path + extensions)
    return None


class PersistingDataset(torch.utils.data.Dataset, ABC):
    def __init__(self, persist_while_fetching: bool = True,
                 video_transform: Compose = None, audio_transform: Compose = None,
                 text_transform: Compose = None, eeg_transform: Compose = None):
        self.persist_while_fetching = persist_while_fetching
        self.extensions = dict()

        self.video_transform: Compose = video_transform
        self.audio_transform: Compose = audio_transform
        self.text_transform: Compose = text_transform
        self.eeg_transform: Compose = eeg_transform

    @abstractmethod
    def persist_on_fetch(self, tensor: torch.Tensor):
        pass

    @abstractmethod
    def get_entry(self, index: int):
        pass


class MediaBasedDataset(PersistingDataset, ABC):
    def __init__(self, persist_while_fetching: bool = True,
                 video_transform: Compose = None, audio_transform: Compose = None,
                 text_transform: Compose = None, eeg_transform: Compose = None, train: bool = True):
        """
        Dataset with the assumption that the audio, video and text data is partitioned in various files.
        :param persist_while_fetching:
        :param video_transform:
        :param audio_transform:
        :param text_transform:
        :param eeg_transform:
        """
        super().__init__(persist_while_fetching)
        self.train = train

        self.video_transform: Compose = video_transform
        self.audio_transform: Compose = audio_transform
        self.text_transform: Compose = text_transform
        self.eeg_transform: Compose = eeg_transform

        # Template.
        media_path = Path(self.get_media(0))
        # Contains the sets of extensions to check
        self.extensions = dict()
        self.text_enabled, self.extensions["text"] = check_extension(media_path, text_extensions())
        self.video_enabled, self.extensions["video"] = check_extension(media_path, video_extensions())
        self.audio_enabled, self.extensions["audio"] = check_extension(media_path, audio_extensions())

    @abstractmethod
    def get_media(self, idx: int) -> str:
        pass

    def get_video(self, idx: int) -> None | torch.Tensor:
        if not self.video_enabled:
            return None

        media_path = self.get_media(idx)
        path = get_filepath(media_path, self.extensions["video"])
        vd = extract_frames(cv2.VideoCapture(path))
        if self.video_transform is not None:
            pre, vd = self.video_transform(vd, train=self.train, return_both=True)
        # todo adesso usa pre e mandalo giu
        return vd if isinstance(vd, torch.Tensor) else torch.as_tensor(vd)

    def get_audio(self, idx: int) -> None | torch.Tensor:
        if not self.audio_enabled:
            return None

        media_path = self.get_media(idx)
        path = get_filepath(media_path, self.extensions["audio"])
        waveform, sr = torchaudio.load(path)

        ad = waveform
        if self.audio_transform is not None:
            pre, ad = self.audio_transform([waveform, sr], train=self.train, return_both=True)
        return ad if isinstance(ad, torch.Tensor) else torch.as_tensor(ad)

    def get_text(self, idx: int) -> None | torch.Tensor:
        if not self.text_enabled or self.text_transform is None:
            return None

        media_path = self.get_media(idx)
        path = get_filepath(media_path, self.extensions["text"])
        with open(path) as f:
            td = f.read()

        if self.text_transform is not None:
            pre, td = self.text_transform(td, train=self.train, return_both=True)
        return torch.as_tensor(td)

    @abstractmethod
    def get_eeg(self, idx: int) -> None | torch.Tensor:
        pass

    def __getitem__(self, idx: int) -> tuple[DatasetRecord, None | torch.Tensor]:
        cached = self.retrieve_from_persistent(idx)
        if cached is not None:
            return cached, None

        record = DatasetRecord(
            eeg=self.get_eeg(idx),
            video=self.get_video(idx),
            audio=self.get_audio(idx),
            text=self.get_text(idx),
            present=None
        )

        present = [record.eeg is not None, record.video is not None, record.audio is not None, record.text is not None]
        record.present = torch.tensor(present, dtype=torch.bool)

        if self.persist_while_fetching:
            self.persist_on_fetch(record)

        return record, None

    @abstractmethod
    def persist_on_fetch(self, record: DatasetRecord):
        pass

    @abstractmethod
    def retrieve_from_persistent(self, idx: int) -> DatasetRecord:
        """
        Returns a dataset record by index if it finds. Else returns None.
        :param idx:
        :return:
        """
        pass

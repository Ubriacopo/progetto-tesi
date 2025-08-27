import dataclasses
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import pandas as pd
import torch
import torchaudio
from moviepy import ImageSequenceClip

from common.data.extensions import text_extensions, video_extensions, audio_extensions
from common.data.sampler import SamplingDescriptor
from common.data.transform import Compose
from common.data.video.utils import extract_frames


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


# todo con consapevolezza di dover gestire file di vario tipo cached diventa diverso.
#   -> salvo su file diversi video/audio/testo con formati coerenti alle loro necessità
class CachedMediaBasedDataset(torch.utils.data.Dataset, ABC):
    """
    Media based data. Medias are processed from files (could be overlapping by overriding the logic to get the data).
    For now, it simply loads video as frames, audio as wave + sr, transcript as text and eeg data by end defined logic.
    Being persisting once a record is called (in use_cache mode) it is persisted somewhere defined by the child class.
    All entries to be stored are assumed to be instanced as torch.Tensor
    """

    def __init__(self, use_cache: bool = True, train: bool = True, video_transform: Compose = None,
                 audio_transform: Compose = None, text_transform: Compose = None, eeg_transform: Compose = None):
        super().__init__()
        self.use_cache: bool = use_cache

        # We work with these kinds of data.
        self.vid_transform: Optional[Compose] = video_transform
        self.aud_transform: Optional[Compose] = audio_transform
        self.txt_transform: Optional[Compose] = text_transform
        assert eeg_transform is not None, "EEG transform must be defined"
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
    def get_cached_path(self, index: int) -> str:
        """
        Returns path of the cached file. (After pre-processing)

        :param index: Index of the cached file
        :return: Path of the cached file
        """
        pass

    @abstractmethod
    def get_eeg(self, idx: int, cached: bool = False) -> Optional[torch.Tensor]:
        pass

    @abstractmethod
    def element_is_cached(self, idx: int) -> bool:
        raise NotImplementedError()

    @abstractmethod
    def register_in_cache(self, idx: int) -> None:
        raise NotImplementedError()

    def get_video(self, idx: int, cached: bool = False) -> list[torch.Tensor] | None:
        if not self.vid_enabled:
            return None

        to_cache = self.use_cache and not cached

        media_path = self.get_media_path(idx) if not cached else self.get_cached_path(idx)
        path = get_filepath(media_path, self.extensions["video"])

        cap = cv2.VideoCapture(path)
        original_fps = cap.get(cv2.CAP_PROP_FPS)

        vd = extract_frames(cap)
        vd = self.vid_transform(vd, train=self.train, return_both=to_cache)

        # TODO:
        # What if pre is empty pipeline? Well, would you like to know! I think this whole pre should just be dropped and done elsewhere.
        # Compose remains as post load processing. (aug + postprocess)
        if to_cache:
            vd, cache_vd = vd

            # TODO: Verifica + come faccio ad ottenere mapped fps da transform? Mi faccio passare un object di meta?
            clip = ImageSequenceClip(cache_vd, fps=original_fps)
            extension = next(self.extensions["video"])
            clip.write_videofile(self.get_cached_path(idx) + extension)

        return vd

    def get_audio(self, idx: int, cached: bool = False) -> Optional[tuple[torch.Tensor, int]]:
        if not self.aud_enabled:
            return None

        to_cache = self.use_cache and not cached

        media_path = self.get_media_path(idx) if not cached else self.get_cached_path(idx)
        path = get_filepath(media_path, self.extensions["audio"])

        ad = torchaudio.load(path)
        ad = self.aud_transform(ad[0], train=self.train, return_both=to_cache)

        if to_cache:
            ad, cache_vd = ad

            extension = next(self.extensions["audio"])
            torchaudio.save(self.get_cached_path(idx) + extension, ad)

        return ad

    def get_text(self, idx: int, cached: bool = False) -> None | str:
        if not self.txt_enabled:
            return None

        to_cache = self.use_cache and not cached

        media_path = self.get_media_path(idx) if not cached else self.get_cached_path(idx)
        path = get_filepath(media_path, self.extensions["text"])

        with open(path) as f:
            td = f.read()

        td = self.txt_transform(td, train=self.train, return_both=to_cache)

        if to_cache:
            td, cache_vd = td

            extension = next(self.extensions["text"])
            open(self.get_cached_path(idx) + extension, "w").write(td)

        return td

    def __getitem__(self, idx: int) -> DatasetRecord:
        # If we use cache and element has been cached previously.
        cached = self.use_cache and self.element_is_cached(idx)
        to_cache = self.use_cache and not cached

        vd = self.get_video(idx, cached)
        ad = self.get_audio(idx, cached)
        td = self.get_text(idx, cached)
        ed = self.get_eeg(idx, cached)

        present = [ed is not None, vd is not None, ad is not None, td is not None]
        record = DatasetRecord(eeg=ed, video=vd, audio=ad, text=td, present=torch.tensor(present, dtype=torch.bool))

        if to_cache:
            # Cache is enabled but the element was not cached yet.
            # We have generated the files, but we'd like a quick reference if it was already cached.
            self.register_in_cache(idx)

        return record


class SimpleMediaBasedDataset(torch.utils.data.Dataset, ABC):
    """
    Media based data. Medias are processed from files (could be overlapping by overriding the logic to get the data).
    For now, it simply loads video as frames, audio as wave + sr, transcript as text and eeg data by end defined logic.
    Being persisting once a record is called (in use_cache mode) it is persisted somewhere defined by the child class.
    All entries to be stored are assumed to be instanced as torch.Tensor
    """

    def __init__(self, train: bool = True, video_transform: Compose = None,
                 audio_transform: Compose = None, text_transform: Compose = None, eeg_transform: Compose = None):
        super().__init__()

        # We work with these kinds of data.
        self.vid_transform: Optional[Compose] = video_transform
        self.aud_transform: Optional[Compose] = audio_transform
        self.txt_transform: Optional[Compose] = text_transform
        assert eeg_transform is not None, "EEG transform must be defined"
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

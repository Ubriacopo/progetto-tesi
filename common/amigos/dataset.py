from pathlib import Path

import numpy as np
import pandas as pd
import torch

from common.amigos.sampler import SamplingDescriptor
from common.ds.dataset import DatasetRecord, MediaBasedDataset
from common.ds.transform import Compose
from common.ds.utils import BoundedMap


# Al momento usabile perchè non persiste. Ma non va bene approccio?
class AMIGOSDataset(MediaBasedDataset):
    def get_persistent_path(self, index: int) -> str:
        e = self.get_entry(index)
        # Store in different experiment ids (To avoid bloating everything).
        path = Path(f"{self.cache_folder}/{e.experiment_id}/{index}")
        path.mkdir(parents=True, exist_ok=True)
        return str(path.resolve())

    def __init__(self, dataset_spec_file: str,
                 cache_folder: str = None,
                 use_cache: bool = False,
                 video_transform: Compose = None,
                 audio_transform: Compose = None,
                 text_transform: Compose = None,
                 eeg_transform: Compose = None):
        # Read the EEG data
        super().__init__(use_cache, video_transform, audio_transform, text_transform, eeg_transform)

        self.descriptor = pd.read_csv(dataset_spec_file)
        self.cache_folder = cache_folder
        files = self.descriptor["data_file"].unique().tolist()
        self.cached_eeg = [np.load(f, allow_pickle=True, mmap_mode='r') for f in files]
        # By using this mini cache we avoid some calls that I honestly don't know if they are fast or slow on pandas.
        self.cached_entries = BoundedMap(100, lru=True)

    def get_eeg(self, idx: int) -> None | torch.Tensor:
        entry = self.get_entry(idx)
        return self.cached_eeg[int(Path(entry.data_file).stem.split("_")[-1])][entry.data_index]

    def get_entry(self, idx: int):
        if idx in self.cached_entries:
            return self.cached_entries[idx]

        record = self.descriptor.iloc[idx].to_dict()
        o = SamplingDescriptor(**record)
        self.cached_entries[idx] = o
        return o

    def get_media(self, idx: int) -> str:
        return self.get_entry(idx).media_path

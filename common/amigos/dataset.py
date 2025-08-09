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
    def __init__(self, dataset_spec_file: str,
                 persist_spec_file: str = None,
                 persist_while_fetching: bool = False,
                 video_transform: Compose = None,
                 audio_transform: Compose = None,
                 text_transform: Compose = None,
                 eeg_transform: Compose = None):
        # Read the EEG data
        super().__init__(persist_while_fetching, video_transform, audio_transform, text_transform, eeg_transform)

        self.descriptor = pd.read_csv(dataset_spec_file)
        self.persist_spec_file = persist_spec_file
        files = self.descriptor["data_file"].unique().tolist()
        self.cached_eeg = [np.load(f, allow_pickle=True, mmap_mode='r') for f in files]
        # By using this mini cache we avoid some calls that I honestly don't know if they are fast or slow on pandas.
        self.cached_entries = BoundedMap(100, lru=True)

    def get_eeg(self, idx: int) -> None | torch.Tensor:
        entry = self.get_entry(idx)
        ed = self.cached_eeg[int(Path(entry.data_file).stem.split("_")[-1])][entry.data_index]
        if self.eeg_transform is not None:
            pre, ed = self.eeg_transform(ed, train=self.train, return_both=True)
        # todo questo potrebbe non essere necessario da transform
        return ed if isinstance(ed, torch.Tensor) else torch.as_tensor(ed)

    def retrieve_from_persistent(self, idx: int) -> DatasetRecord:
        pass

    def persist_on_fetch(self, record: DatasetRecord):
        pass

    # TODO: Cosi non va bene. Io non voglio salvare augmentations. Voglio solo salvare quello che ho processato.
    #       Dovrei avere due transforms?
    #   Prefer shards (LMDB/WebDataset/tar) over millions of tiny files if dataset is large.
    #
    #   Potrei scomporre la classe Compose con due metodi di call (uno transfomr e uno augment)
    #   Questa opzione sembra "papabile" -> potremmo provarci dai.

    def get_entry(self, idx: int):
        if idx in self.cached_entries:
            return self.cached_entries[idx]

        record = self.descriptor.iloc[idx].to_dict()
        o = SamplingDescriptor(**record)
        self.cached_entries[idx] = o
        return o

    def get_media(self, idx: int) -> str:
        return self.get_entry(idx).media_path

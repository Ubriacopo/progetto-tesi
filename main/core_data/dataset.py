from abc import ABC
from pathlib import Path

import pandas as pd
import tensordict
import torch
from torch import device

from main.core_data.data_point import FlexibleDatasetTransformWrapper, FlexibleDatasetPoint


class AgnosticProcessingPdMediaDataset(torch.utils.data.Dataset, ABC):
    def __init__(self, dataset_spec_file: str, pipeline: FlexibleDatasetTransformWrapper):
        super().__init__()
        self.pipeline: FlexibleDatasetTransformWrapper = pipeline
        self.df = pd.read_csv(dataset_spec_file, index_col=False)
        self.df.to_dict(orient="records")

    def __getitem__(self, idx: int):
        data_point = self.df.iloc[idx].to_dict()
        data_point = FlexibleDatasetPoint.from_dict(data_point)
        data_point = self.pipeline.call(data_point)
        return data_point

    def len(self):
        return len(self.df)


class FlexibleEmbeddingsSpecMediaDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_spec_file: str, selected_device: device = None, cache_in_ram: bool = False):
        self.device = selected_device
        if selected_device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.base_path: str = str(Path(dataset_spec_file).parent)
        self.df = pd.read_csv(dataset_spec_file, index_col=False)

        # TODO In futuro supportare l'opzione
        self.cache_in_ram: bool = cache_in_ram
        self.ram_cache = dict()

    def __getitem__(self, idx: int):
        # Descriptor.
        sample = self.df.iloc[idx].to_dict()
        inner_idx, eid, segment = sample["index"], sample["eid"], sample["segment"]
        o = tensordict.load_memmap(self.base_path + "/" + eid)
        return o[inner_idx]

    def __len__(self):
        return len(self.df)
from abc import ABC
from pathlib import Path

import pandas as pd
import torch
from torch import device

from common.data.data_point import FlexibleDatasetTransformWrapper, FlexibleDatasetPoint


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

        self.cache_in_ram: bool = cache_in_ram
        self.ram_cache = dict()

    def extract_entry(self, dictionary: dict | torch.Tensor, return_dictionary: dict, index: int) -> dict:
        """
        Recursively extract entries from nested dictionaries. (Data is partitioned by type and not by sample).

        :param dictionary:
        :param return_dictionary:
        :param index:
        :return:
        """
        for key, value in dictionary.items():
            # Base case.
            if isinstance(value, torch.Tensor):
                return_dictionary[key] = value[index].squeeze().to(self.device)
            # Decompose other dictionaries when met.
            if isinstance(value, dict):
                return_dictionary[key] = self.extract_entry(value, {}, index)

        return return_dictionary

    def __getitem__(self, idx: int):
        # Descriptor.
        sample = self.df.iloc[idx].to_dict()
        inner_idx, eid, segment = sample["index"], sample["eid"], sample["segment"]
        if self.cache_in_ram:
            # TODO empty cache if saturated. and see if it is good.
            self.ram_cache[eid] = torch.load(self.base_path + "/" + eid + ".pt", mmap=True, map_location='cpu')
            o = self.ram_cache[eid]
        else:
            o = torch.load(self.base_path + "/" + eid + ".pt", mmap=True, map_location='cpu')

        # for key in o take the tensor at position inner_idx
        item = self.extract_entry(o, {}, inner_idx)
        return item

    def __len__(self):
        return len(self.df)


class AgnosticMaskedEmbeddingsReadyPdSpecMediaDataset(FlexibleEmbeddingsSpecMediaDataset):
    # TODO vedi se serve.
    def __init__(self, modalities: set[str],
                 dataset_spec_file: str, selected_device: device = None, cache_in_ram: bool = False):
        super().__init__(dataset_spec_file, selected_device, cache_in_ram)
        self.modalities = modalities

    def __getitem__(self, idx: int):
        item = super().__getitem__(idx)
        # Create a mask
        item["mod_mask"] = torch.ones(len(item.keys()))

        for modality in self.modalities:
            if modality not in item:
                item[modality] = torch.zeros(0)
            item["mod_mask"] = item[modality]


class KDAgnosticEmbeddingsReadyPdSpecMediaDataset(torch.utils.data.Dataset):
    pass  # todo vedere di fare.
    # TODO: Mi serve? Ho i dati gia pronti mi basta usare stesso seed su Torch Dataloader. e averne 1 per modello.
    #           Fino a idee contrarie che possono creare problemi mi sembra ragionevole e facile evitare KdDataset

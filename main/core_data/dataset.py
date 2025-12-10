import dataclasses
from abc import ABC
from pathlib import Path
from typing import Tuple

import pandas as pd
import tensordict
import torch
from tensordict import TensorDict
from torch import device

from main.core_data.data_point import FlexibleDatasetTransformWrapper, FlexibleDatasetPoint
from main.core_data.media.assessment.assessment import Assessment
from main.core_data.media.audio import Audio
from main.core_data.media.ecg import ECG
from main.core_data.media.eeg import EEG
from main.core_data.media.text import Text
from main.core_data.media.video import Video
from main.utils.data import MaskedValue
from main.utils.logging import make_logger


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


@dataclasses.dataclass
class RequiredKey:
    key: str
    shape: Tuple[int, ...]
    mask_shape: Tuple[int, ...]
    cannot_miss: bool = False


class FlexibleEmbeddingsSpecMediaDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_spec_file: str, required_keys: list[RequiredKey], main_key: str,
                 selected_device: device = None, cache_in_ram: bool = False):
        """"


        :param dataset_spec_file:
        :param required_keys:
        :param selected_device:
        :param cache_in_ram:
        """

        self.logger = make_logger(self.__class__.__name__)

        self.device = selected_device
        if selected_device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.base_path: str = str(Path(dataset_spec_file).parent)
        self.df = pd.read_csv(dataset_spec_file, index_col=False)

        # TODO In futuro supportare l'opzione
        self.cache_in_ram: bool = cache_in_ram
        self.ram_cache = dict()

        self.required_keys: list[RequiredKey] = required_keys
        self.main_key: str = main_key
        if not main_key in map(lambda x: x.key, self.required_keys):
            raise ValueError(f"main_key={main_key} not in required_keys={required_keys}")

    def __getitem__(self, idx: int):
        try:
            sample = self.df.iloc[idx].to_dict()
            inner_idx, eid, segment = sample["index"], sample["eid"], sample["segment"]
            o = tensordict.load_memmap(self.base_path + "/" + str(eid))

            batch_size = o[self.main_key].batch_size
            for k in self.required_keys:
                # Special hand crafted case for Assessment. TODO: Move elsewhere.
                if k.key == Assessment.modality_code() and k.key in o:
                    # noinspection PyTypeChecker
                    o[k.key] = TensorDict(
                        MaskedValue(data=o[k.key], mask=torch.ones((*batch_size, *k.mask_shape), dtype=torch.bool)),
                        batch_size=batch_size
                    )
                elif isinstance(k, RequiredKey) and k.key not in o:
                    # noinspection PyTypeChecker
                    default = TensorDict(
                        MaskedValue(data=torch.zeros((*batch_size, *k.shape), dtype=torch.float32),
                                    mask=torch.zeros((*batch_size, *k.mask_shape), dtype=torch.bool)),
                        batch_size=batch_size
                    )

                    o.setdefault(k.key, default)

            return o[inner_idx]
        except Exception as e:
            raise e

    def __len__(self):
        return len(self.df)

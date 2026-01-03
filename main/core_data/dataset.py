from __future__ import annotations
import dataclasses
from abc import ABC
from pathlib import Path
from typing import Tuple, Optional, Iterator, List

import pandas as pd
import tensordict
import torch
from tensordict import TensorDict
from torch import device
from torch.utils.data import Sampler, Subset

from main.core_data.data_point import FlexibleDatasetTransformWrapper, FlexibleDatasetPoint
from main.core_data.media.assessment.assessment import Assessment
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


# TODO change sampling instead of Sample uniformly from all samples across all datasets to
#       First choose a dataset, then sample a batch from it
# Single-dataset batches (simplest, very effective)
# Each batch comes from one dataset only, dataset chosen uniformly.
# This is often best with contrastive / SigLIP losses.
class FlexibleEmbeddingsSpecMediaDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_spec_file: str, required_keys: list[RequiredKey], main_key: str,
                 squeeze_mask: bool = False, selected_device: device = None, cache_in_ram: bool = False):
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

        self.squeeze_mask = squeeze_mask

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

                    if self.squeeze_mask:
                        default['mask'] = default['mask'].squeeze()

                    o.setdefault(k.key, default)

            return o[inner_idx]
        except Exception as e:
            raise e

    def __len__(self):
        return len(self.df)


# So we did it to make the fusion/KD training signal cleaner and more stable, and to reduce unintended bias from batch composition
class MultiDataset(torch.utils.data.Dataset):
    def __init__(self, datasets: list[torch.utils.data.Dataset]):
        super().__init__()
        self.logger = make_logger(self.__class__.__name__)
        self.datasets: list[torch.utils.data.Dataset] = datasets
        self.dataset_offsets: list[int] = []

        current_offset = 0
        for dataset in self.datasets:
            self.dataset_offsets.append(current_offset)
            current_offset += len(dataset)

        self.total_len = current_offset

    def __len__(self):
        return self.total_len

    def __getitem__(self, idx: int):
        # Iterate backards
        for ds_id in range(len(self.datasets) - 1, -1, -1):
            if idx >= self.dataset_offsets[ds_id]:
                return self.datasets[ds_id][idx - self.dataset_offsets[ds_id]]

        raise IndexError("Element not found in datasets collection")

    def split(self, train: float, validation: float = .0, seed: int = 1) -> \
            tuple[MultiDataset, MultiDataset, MultiDataset] | tuple[MultiDataset, MultiDataset]:
        generator = torch.Generator().manual_seed(seed)

        if train == 0:
            self.logger.error(f"Cannot make train split at 0%. (train={train}, validation={validation}).")
            raise ValueError("Cannot make train split at 0%. (train={train}, validation={validation}).")

        train_split, val_split, test_split = [], [], []
        for dataset in self.datasets:
            n = len(dataset)
            permutation = torch.randperm(n, generator=generator).tolist()

            n_train = int(train * n)
            train_split.append(Subset(dataset, permutation[:n_train]))
            test_start = n_train  # Test split starts from n_train

            if validation > 0:
                val_split.append(Subset(dataset, permutation[n_train:n_train + int(validation * n)]))
                test_start += int(validation * n)  # Validation split exists thus we have to update start point for test

            test_split.append(Subset(dataset, permutation[test_start:]))

        if len(val_split) == 0:
            self.logger.info(f"Split dataset in: {train}/{1 - train}")
            # Do not return the Validation split in case it is empty
            return MultiDataset(train_split), MultiDataset(test_split)

        self.logger.info(f"Split dataset in: {train}/{validation}/{1 - validation - train}")
        return MultiDataset(train_split), MultiDataset(val_split), MultiDataset(test_split)

    @property
    def dataset_ranges(self):
        # Returns list of (start, length) global index ranges per dataset
        return [(self.dataset_offsets[i], len(self.datasets[i])) for i in range(len(self.dataset_offsets))]


@dataclasses.dataclass
class QueueState:
    perm: torch.Tensor  # permutation of [0..length-1]
    ptr: int  # next position to read

# TODO document
class MultiDatasetQueueBatchSampler(Sampler[list[int]]):
    def __init__(self, multi: MultiDataset, batch_size: int,
                 batches_per_epoch: int, alpha=.0, generator: torch.Generator | None = None):
        super().__init__()
        # Initialize the parameters
        self.dataset: MultiDataset = multi
        self.batch_size: int = batch_size
        self.batches_per_epoch: int = batches_per_epoch
        self.gen = generator if generator is not None else torch.Generator()

        ranges = multi.dataset_ranges
        self.starts = torch.tensor([s for s, l in ranges], dtype=torch.long)
        self.lengths = torch.tensor([l for s, l in ranges], dtype=torch.long)
        if (self.lengths < self.batch_size).any():
            bad = torch.nonzero(self.lengths < self.batch_size, as_tuple=False).flatten().tolist()
            raise ValueError(
                f"Datasets smaller than batch_size={self.batch_size}: {bad} (lengths={self.lengths.tolist()})"
            )

        # Presence of each dataset in the random draws
        weights = self.lengths.float().pow(alpha)
        self.probs = weights / weights.sum()
        self._states: list[QueueState] = []
        for length in self.lengths.tolist():
            perm = torch.randperm(length, generator=self.gen)
            self._states.append(QueueState(perm=perm, ptr=0))

    def __len__(self) -> int:
        return self.batches_per_epoch

    def _reshuffle(self, d: int) -> None:
        length = int(self.lengths[d].item())
        self._states[d].perm = torch.randperm(length, generator=self.gen)
        self._states[d].ptr = 0

    def _next_local_batch(self, dataset_idx: int) -> torch.Tensor:
        current_state = self._states[dataset_idx]
        length = int(self.lengths[dataset_idx].item())

        # If enough contiguous items left in queue, take them.
        if current_state.ptr + self.batch_size <= length:
            out = current_state.perm[current_state.ptr: current_state.ptr + self.batch_size]
            current_state.ptr += self.batch_size
            return out

        # Not enough left.
        # Discard tail and reshuffle for a fresh full batch
        self._reshuffle(dataset_idx)
        current_state = self._states[dataset_idx]

        out = current_state.perm[0:self.batch_size]
        current_state.ptr = self.batch_size

        return out

    def __iter__(self) -> Iterator[List[int]]:
        for _ in range(self.batches_per_epoch):
            d = int(torch.multinomial(self.probs, 1, replacement=True, generator=self.gen).item())
            start = int(self.starts[d].item())

            local = self._next_local_batch(d)
            yield (local + start).tolist()


class DatasetFirstBatchSampler(Sampler[list[int]]):
    def __init__(self, multi: MultiDataset, batch_size: int,
                 batches_per_epoch: int, alpha=.0, generator: torch.Generator | None = None):
        super().__init__()
        # Initialize the parameters
        self.dataset: MultiDataset = multi
        self.batch_size: int = batch_size
        self.batches_per_epoch: int = batches_per_epoch
        self.gen = generator if generator is not None else torch.Generator()

        ranges = multi.dataset_ranges
        self.starts = torch.tensor([s for s, l in ranges], dtype=torch.long)
        self.lengths = torch.tensor([l for s, l in ranges], dtype=torch.long)

        weights = (self.lengths.float() ** alpha)
        self.probs = weights / weights.sum()

    def __len__(self):
        return self.batches_per_epoch

    def __iter__(self):
        for _ in range(self.batches_per_epoch):
            dataset_id = torch.multinomial(self.probs, 1, replacement=True, generator=self.gen).item()
            start = self.starts[dataset_id].item()
            length = self.lengths[dataset_id].item()

            if length < self.batch_size:
                raise ValueError(
                    f"Dataset {dataset_id} has length {length}, " f"which is smaller than batch_size={self.batch_size}."
                    f"Thus we cannot sample a unique batch."
                )

            local = torch.randperm(length, generator=self.gen)[:self.batch_size].tolist()
            yield [start + j for j in local]


# todo rename
class SequentialPerDatasetBatchSampler(Sampler[list[int]]):
    def __init__(self, multi: MultiDataset, batch_size: int, drop_last: bool = False):
        super().__init__()

        self.dataset: MultiDataset = multi
        self.batch_size: int = batch_size
        self.drop_last: bool = drop_last
        self.ranges = [(s, l) for s, l in multi.dataset_ranges]

    def __iter__(self):
        for dataset_id, (start, length) in enumerate(self.ranges):
            idx_list = list(range(start, start + length))
            for i in range(0, length, self.batch_size):
                chunk = idx_list[i:i + self.batch_size]
                if len(chunk) < self.batch_size and self.drop_last:
                    continue

                yield chunk

    def __len__(self):
        total = 0

        for start, length in self.ranges:
            q, r = divmod(length, self.batch_size)
            total += q + (0 if (r == 0 or self.drop_last) else 1)

        return total

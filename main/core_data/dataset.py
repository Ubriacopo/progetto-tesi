from __future__ import annotations

import dataclasses
import fcntl
import glob
import json
import os
import shutil
import subprocess
import time
import uuid
from abc import ABC
from functools import lru_cache
from pathlib import Path
from typing import Tuple, Iterator, List, Optional

import h5py
import pandas as pd
import tensordict
import torch
from cachetools import LRUCache
from tensordict import TensorDict
from torch import device
from torch.utils.data import Sampler, Subset
from torch.utils.data import get_worker_info

from main.core_data.data_point import FlexibleDatasetTransformWrapper, FlexibleDatasetPoint
from main.dataset.quantization import Float16ToInt8Quantizer
from main.utils.logging import make_logger


@dataclasses.dataclass
class RequiredKey:
    key: str
    shape: Tuple[int, ...]
    mask_shape: Tuple[int, ...]
    cannot_miss: bool = False


def _worker_cache():
    wi = get_worker_info()
    # each worker has its own Dataset instance, so self-scoped cache is enough
    return


class SimpleDiskLRU:
    def __init__(self, remote_dir: str, local_dir: str, max_gb: int, shard_gb: int = 4):
        self.remote: Path = Path(remote_dir)
        self.local: Path = Path(local_dir)
        # Add the folder if it non existent
        self.local.mkdir(parents=True, exist_ok=True)

        self.max_items = max(1, int((max_gb * (1 << 30)) // (shard_gb * (1 << 30))))
        self.cache = LRUCache(maxsize=self.max_items)

    @staticmethod
    def atomic_copy(src: Path, dst: Path):
        if not src.is_dir():
            raise ValueError(f"Expected directory source, got: {src}")

        dst.parent.mkdir(parents=True, exist_ok=True)
        tmp = dst.parent / (dst.name + f".tmp_{uuid.uuid4().hex}")
        # Copy to TMP
        shutil.copytree(src, tmp, dirs_exist_ok=False)
        # Put into the correct path
        os.replace(tmp, dst)

    def get(self, shard_name: str):
        shard: Path = self.cache.get(shard_name)
        if shard is not None and shard.exists():
            # Touch the existing shard to refresh recency
            self.cache[shard_name] = shard
            return shard

        while len(self.cache) > self.max_items:
            old_name, old_path = self.cache.popitem()  # LRU
            try:
                old_path.unlink()
            except FileNotFoundError:
                pass  # We failed for some reason but as long as we loose it we don't care

        remote = self.remote / shard_name
        local = self.local / shard_name
        if not local.exists():
            self.atomic_copy(remote, local)

        self.cache[shard_name] = local
        return local


@dataclasses.dataclass
class CachableDatasetDescriptor:
    dataset_spec_file: str
    cache_path: Optional[str]


# todo verify
class CachingQuantizedSpecMediaDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_spec_file: str, cache_path: str, selected_device='cpu'):
        self.logger = make_logger(self.__class__.__name__)
        self.device = selected_device

        self.quantizer = Float16ToInt8Quantizer()

        self.base_path: str = str(Path(dataset_spec_file).parent)
        self.cache_path: str = cache_path
        # Read the spec
        self.df = pd.read_csv(dataset_spec_file, index_col=False)
        # If the cache path is the same as the dataset path we are using a cached version directly (AMIGOS, ..)
        # if the cache path is not given also no caching enabled
        self.lru_caching_enabled = self.cache_path is not None and self.base_path != self.cache_path

        # Initialized differently for each worker
        self.disk_lru = None

        self.open_shard_td: Optional[TensorDict] = None
        self.open_shard_name: Optional[str] = None

        self.inner_idx = self.df["index"].to_numpy()
        self.eid_list = self.df["eid"].astype(str).to_numpy()
        self.sharded_eid = self.df["sharded_eid"].to_numpy()
        self.sharded_idx = self.df["sharded_index"].to_numpy()

    @staticmethod
    def _worker_cache_dir(base_cache_dir: str) -> str:
        worker_info = get_worker_info()
        return base_cache_dir if worker_info is None else str(Path(base_cache_dir) / f"w{worker_info.id}")

    def _get_lru(self):
        if self.disk_lru is None and self.lru_caching_enabled:
            worker_info = get_worker_info()
            num_workers = worker_info.num_workers if worker_info else 1

            self.disk_lru = SimpleDiskLRU(
                self.base_path, self._worker_cache_dir(self.cache_path), max(4, 30 // num_workers), shard_gb=4
            )

        # Return the currently set instance (might be None if lru_caching is disabled.
        return self.disk_lru

    def get_shard_td(self, shard_path: Path, shard_name: str):
        # Avoid re-opening memmap for every sample so we store in current memory
        if self.open_shard_name != shard_name:
            self.open_shard_td = tensordict.load_memmap(shard_path)
            # Used for checking if the tensordict is open
            self.open_shard_name = shard_name

        return self.open_shard_td

    def __getitem__(self, item):
        # todo verify this acces
        shard_name = self.sharded_eid[item]
        idx_in_shard = int(self.sharded_idx[item])

        lru = self._get_lru()
        if lru is None:
            shard_path = Path(self.base_path) / str(shard_name)
        else:
            shard_path = lru.get(str(shard_name))

        shard = self.get_shard_td(shard_path, str(shard_name))
        sample = shard[idx_in_shard]
        # In hope to speed up training
        sample.pop("meta", None)
        return sample

    def __len__(self):
        return len(self.df)


@dataclasses.dataclass(frozen=True)
class SampleReference:
    shard_idx: int
    local_idx: int

# todo iterable dataset
class H5KdSourceDataset(torch.utils.data.Dataset):
    def __init__(self, shards_path: str, selected_device="cpu", index_cache_name="index.json"):
        super().__init__()
        self.logger = make_logger(self.__class__.__name__)

        self.device = selected_device
        if self.device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # todo use pathlib
        self.shard_paths: list[str] = sorted(glob.glob(os.path.join(shards_path, "*.h5")))
        if not self.shard_paths:
            raise FileNotFoundError(f"No .h5 shards found in: {shards_path}")

        cache_path = os.path.join(shards_path, index_cache_name)
        self.shard_lengths = self.load_lengths(cache_path)

        self.refs: list[SampleReference] = []
        for s_idx, n in enumerate(self.shard_lengths):
            self.refs.extend(SampleReference(s_idx, i) for i in range(n))

        self._open_files: dict[int, h5py.File] = {}  # key: shard_idx

    # todo meh
    def load_lengths(self, cache_path: str):
        # todo use pandas csv
        if os.path.exists(cache_path):
            with open(cache_path, "r") as f:
                obj = json.load(f)
            if obj.get("shards") == self.shard_paths:
                return obj["lengths"]

        lengths = []
        for p in self.shard_paths:
            with h5py.File(p, "r") as h5:
                # Prefer attr; fallback to dataset length
                n = int(h5.attrs.get("num_samples", 0))
                if n <= 0:
                    # e.g. eid dataset exists
                    n = int(h5["eid"].shape[0]) if "eid" in h5 else 0
                if n <= 0:
                    raise ValueError(f"Could not determine num_samples for shard: {p}")
                lengths.append(n)

        with open(cache_path, "w") as f:
            json.dump({"shards": self.shard_paths, "lengths": lengths}, f)
        return lengths

    def __getitem__(self, idx: int):
        pass

    # todo Buffer shuffle (best compromise):

    def __len__(self) -> int:
        return len(self.refs)


class FlexibleEmbeddingsSpecMediaDatasetSlow(torch.utils.data.Dataset):
    def __init__(self, dataset_spec_file: str, selected_device='cpu'):
        """"


        :param dataset_spec_file:
        :param selected_device:
        :param cache_in_ram:
        """
        super().__init__()
        self.logger = make_logger(self.__class__.__name__)

        self.device = selected_device
        if self.device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.base_path: str = str(Path(dataset_spec_file).parent)
        self.df = pd.read_csv(dataset_spec_file, index_col=False)
        self.inner_idx = self.df["index"].to_numpy()
        self.eid_list = self.df["eid"].astype(str).to_numpy()

    def _load_uncached(self, eid: str):
        td = tensordict.load_memmap(f"{self.base_path}/{eid}")
        # Do “one-time” cleanup here if possible
        td.pop("assessment", None)
        return td

    def __getitem__(self, idx: int):
        try:
            # Questa variante è troppo lenta
            inner_idx = int(self.inner_idx[idx])
            eid = self.eid_list[idx]
            td = tensordict.load_memmap(f"{self.base_path}/{eid}", device=self.device)
            # Do “one-time” cleanup here if possible
            td.pop("assessment", None)
            return td[inner_idx]

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
        bs = self.batch_size
        for start, length in self.ranges:
            end = start + length
            for i in range(start, end, bs):
                chunk_end = min(i + bs, end)
                if chunk_end - i < bs and self.drop_last:
                    continue
                yield list(range(i, chunk_end))

    def __len__(self):
        total = 0

        for start, length in self.ranges:
            q, r = divmod(length, self.batch_size)
            total += q + (0 if (r == 0 or self.drop_last) else 1)

        return total

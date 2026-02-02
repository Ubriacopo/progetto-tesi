from __future__ import annotations

import dataclasses
import glob
import json
import os
import random
import shutil
import uuid
from pathlib import Path
from typing import Tuple, Iterator, Optional, Any, Mapping

import h5py
import numpy as np
import pandas as pd
import tensordict
import torch
from cachetools import LRUCache
from tensordict import TensorDict
from torch.utils.data import Sampler, Subset, IterableDataset
from torch.utils.data import get_worker_info

from main.core_data.media.audio import Audio
from main.core_data.media.ecg import ECG
from main.core_data.media.eeg import EEG
from main.core_data.media.text import Text
from main.core_data.media.video import Video
from main.utils.logging import make_logger


@dataclasses.dataclass
class RequiredKey:
    key: str
    shape: Tuple[int, ...]
    mask_shape: Tuple[int, ...]
    cannot_miss: bool = False


@dataclasses.dataclass
class CachableDatasetDescriptor:
    dataset_path: str
    cache_path: Optional[str]
    dataset_weight: float



class H5KdDataset(IterableDataset):
    def __init__(self, dataset_path: str, prefix: str, batch_size: int, device="cpu", buffer_size: int = 256,
                 block_size: int = 256, seed: int = 42, ignore_paths: list[str] = None):
        super().__init__()
        self.logger = make_logger(self.__class__.__name__)

        self.dataset_path: Path = Path(dataset_path)
        self.prefix: str = prefix

        self.device = device
        self.shard_files = sorted(glob.glob(os.path.join(dataset_path, f"{prefix}*.h5")))
        if not self.shard_files:
            raise FileNotFoundError(f"No .h5 shards found in: {dataset_path} with prefix: {prefix}")

        self.buffer_size: int = buffer_size
        if self.buffer_size <= 0:
            raise ValueError("buffer_size must be > 0")

        self.seed: int = seed
        self.epoch: int = 0

        self.block_size: int = block_size
        self.shard_lengths: list[int] = list(self.load_lengths())
        self.ignore_paths: list[str] = ignore_paths

        self.paths: Optional[list[str]] = None
        self.batch_size: int = batch_size

    def load_lengths(self) -> Iterator[int]:
        for file in self.shard_files:
            with h5py.File(file, "r") as h5:
                yield int(h5.attrs.get("num_samples", 0))

    def data_for_worker(self, g: torch.Generator):
        num_workers = 1 if get_worker_info() is None else get_worker_info().num_workers
        worker_id = 0 if get_worker_info() is None else get_worker_info().id

        perm = torch.randperm(len(self.shard_files), generator=g).tolist()
        files = [self.shard_files[i] for i in perm]
        lengths = [self.shard_lengths[i] for i in perm]

        # If the number of workers is enough to cover all files without overlap.
        if num_workers <= len(files):
            files = files[worker_id:: num_workers]
            lengths = lengths[worker_id:: num_workers]

            for file, length in zip(files, lengths):
                yield file, 0, length
            return

        # We cannot cover all files, so sharing is caring.
        # We make now choices based on file (we look them all up)
        for file, length in zip(files, lengths):
            if length <= 0:
                continue

            num_blocks = (length + self.block_size - 1) // self.block_size
            # If the number of blocks is enough for all workers we divide the blocks.
            if num_blocks >= num_workers:
                for block_start in range(0, length, self.block_size):
                    block_id = block_start // self.block_size
                    if block_id % num_workers == worker_id:
                        yield file, block_start, min(block_start + self.block_size, length)

            # Worst case: blocks have to be split to give info to all workers.
            else:
                step = max(1, length // num_workers)
                start = worker_id * step
                stop = length if worker_id == num_workers - 1 else min(length, (worker_id + 1) * step)

                if start < stop:
                    yield file, start, stop

    def collect_h5_paths(self, h5: h5py.File, ignore_prefixes: list[str] = ("meta/",)):
        if self.paths is not None:
            return self.paths

        self.paths = []

        def visit(name, obj):
            if isinstance(obj, h5py.Dataset) and not any(name.startswith(p) for p in ignore_prefixes):
                self.paths.append(name)

        h5.visititems(visit)
        return self.paths

    @staticmethod
    def read_block(dsets: dict[str, h5py.Dataset], start: int, stop: int):
        output = {}
        for path, dataset in dsets.items():
            arr = dataset[start:stop]
            if isinstance(arr, np.ndarray) and arr.shape and arr.dtype.kind in "iufb":
                if not arr.flags['C_CONTIGUOUS']:
                    arr = np.ascontiguousarray(arr)  # Make contiguous if needed
                arr = torch.from_numpy(arr)
            output[path] = arr

        return output

    def iter_shard_blocks(self, h5: h5py.File, start: int, stop: int):
        n = int(h5.attrs.get("num_samples", 0)) or int(h5["meta/eid"].shape[0])
        start, stop = max(0, int(start)), min(int(stop), n)
        if stop <= start:
            self.logger.warning("Start precedes stop? Are you sure?")
            return

        paths = self.collect_h5_paths(h5, ignore_prefixes=["meta/"])
        dsets = {p: h5[p] for p in paths}

        for block_start in range(start, stop, self.block_size):
            block_stop = min(block_start + self.block_size, stop)
            # dict[str, np/torch array], len = block_stop-block_start
            yield self.read_block(dsets, block_start, block_stop)

    @staticmethod
    def add_block(buffered_samples: int, block: dict[Any, Any], block_buffer: list, rng: random.Random) -> int:
        block_len = len(next(iter(block.values())))
        # Make a shuffled index order for this block
        perm = list(range(block_len))
        rng.shuffle(perm)

        block_buffer.append([block, perm, 0])
        return buffered_samples + block_len

    @staticmethod
    def pop_random_sample(buffered_samples: int, block_buffer: list, rng: random.Random) -> tuple[dict[Any, Any], int]:
        idx = rng.randrange(len(block_buffer))
        block, perm, current = block_buffer[idx]
        i = perm[current]

        # Update or remove the block entry
        if current + 1 >= len(perm):
            block_buffer.pop(idx)
        else:
            block_buffer[idx][2] = current + 1

        sample = {k: v[i] for k, v in block.items()}
        return sample, buffered_samples - 1

    def pop_random_batch(self, buffered_samples: int, block_buffer: list, rng: random.Random):
        output = []
        for batch_element in range(self.batch_size):
            idx = rng.randrange(len(block_buffer))
            block, perm, current = block_buffer[idx]

            i = perm[current]
            if current + 1 >= len(perm):
                block_buffer.pop(idx)
            else:
                block_buffer[idx][2] = current + 1

            sample = {k: v[i] for k, v in block.items()}
            output.append(sample)  # breaks storage sharing

            buffered_samples -= 1
            if not block_buffer:
                break

        return output, buffered_samples

    def __iter__(self):
        # After various rewrites this gives best performance yet.
        worker_info = get_worker_info()
        worker_id = 0 if worker_info is None else worker_info.id

        global_g = torch.Generator()
        global_g.manual_seed(self.seed + self.epoch)
        rng = random.Random(self.seed + self.epoch + 10000 * worker_id)

        warmup: int = min(128, self.buffer_size)
        # We buffer blocks not samples. Each entry is of the structure: [block_dict, perm_list, cursor]
        block_buffer = []

        # Total remaining samples across all buffered blocks
        buffered_samples = 0
        target = self.block_size * 2
        for shard_path, start, stop in self.data_for_worker(global_g):
            with h5py.File(str(shard_path), "r") as h5:
                for block in self.iter_shard_blocks(h5, start=start, stop=stop):
                    buffered_samples = self.add_block(buffered_samples, block, block_buffer, rng)

                    # Warmup: ensure at least warmup samples are buffered before yielding
                    if buffered_samples < warmup:
                        continue

                    while buffered_samples >= target and block_buffer:
                        sample, buffered_samples = self.pop_random_batch(buffered_samples, block_buffer, rng)
                        yield sample

        # Flush remaining buffered samples
        while block_buffer:
            sample, buffered_samples = self.pop_random_batch(buffered_samples, block_buffer, rng)

            yield sample


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
        self.df = pd.read_csv(dataset_spec_file, index_col=False, )
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


class RoundRobinBatchMultiDataset(IterableDataset):
    def __init__(self, datasets: list[IterableDataset], weights, seed: int, batch_size: int):
        super().__init__()
        self.logger = make_logger(self.__class__.__name__)
        self.datasets: list[IterableDataset] = datasets

        w = torch.tensor(weights, dtype=torch.float)
        self.weights = w

        self.epoch = 0
        self.seed = seed

        self.batch_size: int = batch_size

    def set_epoch(self, epoch: int):
        self.epoch = int(epoch)
        for dataset in self.datasets:
            if hasattr(dataset, "set_epoch"):
                dataset.set_epoch(epoch)

    def __iter__(self):
        worker_info = get_worker_info()

        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch + (0 if worker_info is None else 10000 * worker_info.id))

        iters = [iter(ds) for ds in self.datasets]
        dead = [0] * len(iters)  # Count consecutive failures per dataset
        while True:
            k = int(torch.multinomial(self.weights, num_samples=1, replacement=True, generator=g).item())
            for _ in range(24):
                try:
                    yield next(iters[k])
                    dead[k] = 0

                except StopIteration:
                    dead[k] += 1
                    if dead[k] > 5:
                        raise RuntimeError(f"Dataset {k} keeps stopping; is it empty?")
                    # Restart that dataset (cycle)
                    iters[k] = iter(self.datasets[k])
                    break

                # yield TensorDict.stack(batch, dim=0)


# todo make version for multiIterableDataset
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

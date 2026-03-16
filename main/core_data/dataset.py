from __future__ import annotations

import dataclasses
import glob
import os
import random
from pathlib import Path
from typing import Tuple, Iterator, Optional, Any

import h5py
import numpy as np
import pandas as pd
import tensordict
import torch
from torch.utils.data import Subset, IterableDataset
from torch.utils.data import get_worker_info

from main.utils.logging import make_logger


@dataclasses.dataclass
class RequiredKey:
    key: str
    shape: Tuple[int, ...]
    mask_shape: Tuple[int, ...]
    cannot_miss: bool = False


@dataclasses.dataclass
class CacheableDatasetDescriptor:
    dataset_path: str
    cache_path: Optional[str]
    dataset_weight: float


import time


class Timer:
    def __init__(self):
        self.totals = {}

    def add(self, name, dt):
        self.totals[name] = self.totals.get(name, 0.0) + dt

    def report(self):
        for k, v in self.totals.items():
            print(f"{k:20s}: {v:.3f} sec")


class H5KdDataset(IterableDataset):
    def __init__(self, dataset_path: str, prefix: str, batch_size: int, buffer_size: int, block_size: int = 256,
                 seed: int = 42, shuffle: bool = True, iterator_id: int = None, limit_data: float = None):
        """

        :param dataset_path:
        :param prefix:
        :param batch_size:
        :param buffer_size:
        :param block_size:
        :param seed:
        :param shuffle:
        :param iterator_id:
        :param limit_data: Fraction of the dataset to load
        """
        self.logger = make_logger(self.__class__.__name__)

        self.dataset_path: Path = Path(dataset_path)
        self.shard_files = sorted(glob.glob(os.path.join(dataset_path, f"{prefix}*.h5")))
        if not self.shard_files:
            raise FileNotFoundError(f"No .h5 shards found in: {dataset_path} with prefix: {prefix}")

        self.shard_lengths: list[int] = list(self.load_lengths())
        # If we want we can use a subset of data, it is handled by expliciting the % of desired data
        if limit_data is not None:
            target_samples = sum(self.shard_lengths) * limit_data
            shard_pairs = list(zip(self.shard_files, self.shard_lengths))

            # Shuffle for better mixin
            rng = random.Random(seed)
            rng.shuffle(shard_pairs)

            selected_files, selected_lengths = [], []
            collected = 0

            for shard_file, shard_length in shard_pairs:
                selected_files.append(shard_file)
                selected_lengths.append(shard_length)
                # Increase by collected shards
                collected += shard_length

                if collected >= target_samples:
                    break

            self.shard_files = selected_files
            self.shard_lengths = selected_lengths

        self.batch_size: int = batch_size
        self.block_size: int = block_size

        self.seed: int = seed
        self.paths: Optional[list[str]] = None
        self.buffer_size: int = buffer_size
        self.shuffle: bool = shuffle
        # To count the iter
        self.iterator_id: int = iterator_id if iterator_id is not None else 0

        self.timer = Timer()

    def load_lengths(self) -> Iterator[int]:
        for file in self.shard_files:
            with h5py.File(file, "r") as h5:
                yield int(h5.attrs.get("num_samples", 0))

    def collect_h5_paths(self, h5: h5py.File, ignore_prefixes: list[str] = ("meta/",)):
        # todo keep ds id
        if self.paths is None:
            # Initialize paths only once per worker
            self.paths = []

            def visit(name: str, obj: h5py.Dataset):
                if isinstance(obj, h5py.Dataset) and not any(name.startswith(p) for p in ignore_prefixes):
                    self.paths.append(name)

            h5.visititems(visit)

        return self.paths

    def data(self, generator: Optional[torch.Generator] = None) -> Iterator[Any]:
        num_workers = 1 if get_worker_info() is None else get_worker_info().num_workers
        worker_id = 0 if get_worker_info() is None else get_worker_info().id

        n = len(self.shard_files)
        if self.shuffle and generator is not None:
            perm = torch.randperm(len(self.shard_files), generator=generator).tolist()
        else:
            perm = list(range(n))

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

    def add_block(self, buffered_samples: int, block: dict[Any, Any], block_buffer: list, rng: random.Random = None):
        block_len = len(next(iter(block.values())))
        # Make a shuffled index order for this block if shuffling is enabled
        perm = None
        if rng is not None:
            perm = torch.randperm(block_len)  # CPU tensor

        block_buffer.append([block, perm, 0])
        return buffered_samples + block_len

    def pop_batch(self, buffered_samples: int, block_buffer: list, rng: random.Random = None):
        """Return dict[str, Tensor[B,...]] built from multiple blocks."""
        if not block_buffer or buffered_samples <= 0:
            return None, buffered_samples

        parts: list[dict[str, torch.Tensor]] = []
        need: int = self.batch_size

        # Fill batch from multiple blocks
        while need > 0 and block_buffer and buffered_samples > 0:
            bidx: int | None = None
            if rng is not None:
                random_selection = rng.randrange(buffered_samples)

                accumulator = 0
                for idx, (block, perm, cursor) in enumerate(block_buffer):
                    block_len = len(perm) if perm is not None else len(next(iter(block.values())))
                    available = block_len - cursor

                    if available <= 0:
                        continue

                    accumulator += available
                    if random_selection < accumulator:
                        bidx = idx
                        break

            else:
                for idx, (block, perm, cursor) in enumerate(block_buffer):
                    block_len = len(perm) if perm is not None else len(next(iter(block.values())))
                    if block_len - cursor > 0:
                        bidx = idx
                        break

            if bidx is None:
                break

            block, perm, cursor = block_buffer[bidx]
            block_len = len(perm) if perm is not None else len(next(iter(block.values())))
            avail = block_len - cursor

            if avail <= 0:
                block_buffer.pop(bidx)
                continue

            take = min(need, avail)
            sel = perm[cursor:cursor + take] if perm is not None else slice(cursor, cursor + take)

            # advance cursor / retire block if done
            cursor += take
            if cursor >= block_len:
                block_buffer.pop(bidx)
            else:
                block_buffer[bidx][2] = cursor

            parts.append({k: v[sel] for k, v in block.items()})

            buffered_samples -= take
            need -= take

        if not parts:
            return None, buffered_samples

        # concat along batch dim
        out = {k: torch.cat([p[k] for p in parts], dim=0) for k in parts[0].keys()}

        # If you require exactly batch_size always, you can drop incomplete batches:
        # if out[next(iter(out))].size(0) != self.batch_size: return None, buffered_samples

        return out, buffered_samples

    def strong_pop_batch(self, buffered_samples: int, block_buffer: list, rng: random.Random = None):
        output: list[dict[str, torch.Tensor]] = []
        for batch_element in range(self.batch_size):
            if not block_buffer:
                break

            idx = rng.randrange(len(block_buffer)) if rng is not None else 0
            block, perm, cursor = block_buffer[idx]

            i = perm[cursor] if perm is not None else cursor
            cursor += 1

            target_measure = len(perm) if perm is not None else len(next(iter(block.values())))
            done = cursor >= target_measure

            if done:
                block_buffer.pop(idx)
            else:
                block_buffer[idx][2] = cursor

            sample = {k: v[i] for k, v in block.items()}
            output.append(sample)  # breaks storage sharing
            buffered_samples -= 1

        out = {k: torch.cat([p[k] for p in output], dim=0) for k in output[0].keys()}
        return out, buffered_samples

    def read_block(self, dsets: dict[str, h5py.Dataset], start: int, stop: int):
        output = {}

        for path, dataset in dsets.items():
            t0 = time.perf_counter()
            arr = dataset[start:stop]
            dt = time.perf_counter() - t0
            self.timer.add("allocate_tensor_" + path, dt)

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

            t0 = time.perf_counter()
            block = self.read_block(dsets, block_start, block_stop)
            self.timer.add("read_block", time.perf_counter() - t0)
            yield block

    def __iter__(self):
        # We still presume one worker only but this will come in handy later.
        worker_info = get_worker_info()
        worker_id = 0 if worker_info is None else worker_info.id

        iterator_id = self.iterator_id
        self.iterator_id += 1

        epoch_seed = (self.seed + iterator_id) & 0xFFFFFFFF

        global_g = torch.Generator()
        global_g.manual_seed(epoch_seed)

        rng: Optional[random.Random] = None
        if self.shuffle:
            # 0x9E3779B9 is derived from the fractional part of the Golden Ration and used in hash functions to decorrelate nearby ints
            # While 100003 is a large prime that prevents the worker from getting seeds that differ only by +1
            rng = random.Random((epoch_seed + 0x9E3779B9 + worker_id * 1000003) & 0xFFFFFFFF)

        warmup: int = self.buffer_size

        # We buffer blocks not samples. Each entry is of the structure: [block_dict, perm_list, cursor]
        block_buffer = []

        buffered_samples = 0
        for shard_path, start, stop in self.data(generator=global_g):
            t0 = time.perf_counter()
            with h5py.File(str(shard_path), "r", locking=False,
                           rdcc_nbytes=64 * 1024 * 1024, rdcc_nslots=100_003, rdcc_w0=0.75, ) as h5:
                self.timer.add("open_file", time.perf_counter() - t0)
                for block in self.iter_shard_blocks(h5, start=start, stop=stop):
                    buffered_samples = self.add_block(buffered_samples, block, block_buffer, rng)
                    # Warmup: ensure at least warmup samples are buffered before yielding
                    if buffered_samples < warmup:
                        continue

                    while buffered_samples >= self.buffer_size and block_buffer:
                        t0 = time.perf_counter()
                        sample, buffered_samples = self.pop_batch(buffered_samples, block_buffer, rng)
                        self.timer.add("pop_batch", time.perf_counter() - t0)
                        yield sample

        # Flush remaining buffered samples
        while block_buffer:
            t0 = time.perf_counter()
            sample, buffered_samples = self.pop_batch(buffered_samples, block_buffer, rng)
            self.timer.add("pop_batch", time.perf_counter() - t0)
            yield sample

    def __len__(self):
        return int(sum(self.shard_lengths) / self.batch_size)  # We return batches stuff so it is smaller.


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
    def __init__(self, datasets: list[IterableDataset], weights, seed: int, consecutive_batches: int):
        """
        RoundRobinBatchMultiDataset is a dataset that switches between source datasets in a round-robin manner.
        Difference to real round-robin is that it actually draws with a probability p what dataset to select next.
        Source datasets once exhausted are re-initialized, this to allow smaller datasets to still be relevant.

        :param datasets: List of source iterable datasets to switch on
        :param weights: Weight of each dataset. This is used to calculate the probability of drawing from a dataset
        :param seed: Random seed
        :param consecutive_batches: How many batches of the same source appear forced in sequence
        """
        super().__init__()
        self.logger = make_logger(self.__class__.__name__)
        self.datasets: list[IterableDataset] = datasets

        w = torch.tensor(weights, dtype=torch.float)
        self.weights = w

        self.seed: int = seed
        self.consecutive_batches: int = consecutive_batches

    def __iter__(self):
        g = torch.Generator()
        seed = (torch.initial_seed() + self.seed) % 2 ** 32
        g.manual_seed(seed)

        iters = [iter(ds) for ds in self.datasets]
        dead = [0] * len(iters)  # Count consecutive failures per dataset
        while True:
            k = int(torch.multinomial(self.weights, num_samples=1, replacement=True, generator=g).item())
            for _ in range(self.consecutive_batches):
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

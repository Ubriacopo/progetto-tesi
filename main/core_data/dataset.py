from __future__ import annotations

import dataclasses
import glob
import json
import os
import random
import shutil
import uuid
from pathlib import Path
from typing import Tuple, Iterator, Optional, Any

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


class H5DataDictExtractor:
    def __init__(self):
        self.state: Optional[dict] = None

    def slice_h5_to_dict(self, h5_obj: h5py.File | h5py.Group | h5py.Dataset, start: int, stop: int):
        self.state = {}
        for key, item in h5_obj.items():
            if isinstance(item, h5py.Group):
                self.state[key] = self.slice_h5_to_dict(item, start, stop)

            elif isinstance(item, h5py.Dataset):
                self.state[key] = item[start:stop]

        return self.state


class H5ModalityShardExtractor:
    def __init__(self, modalities: dict[str, list[str]]):
        # deep structures are not supported as we don't need them
        self.modalities: dict[str, list[str]] = modalities
        self.state: dict = None

    @staticmethod
    def default():
        return H5ModalityShardExtractor({
            Video.modality_code(): ["data", "mask", "scales"],
            Audio.modality_code(): ["data", "mask", "scales"],
            Text.modality_code(): ["data", "mask", "scales"],
            EEG.modality_code(): ["data", "mask", "scales"],
            ECG.modality_code(): ["data", "mask", "scales"]
        })

    def take(self, idx: int):
        out = {}
        # todo misura per vedere performance
        for modality, field_map in self.state.items():
            modality_object = {}
            for field, arr in field_map.items():
                modality_object[field] = arr[idx]
            out[modality] = modality_object

        return out

    def make(self, h5: h5py.File, start: int, end: int) -> dict:
        self.state = {}
        for modality, fields in self.modalities.items():
            if modality not in h5:
                # Modality ignored as isn't there
                continue

            modality_object = {}

            for field in fields:
                if f"{modality}/{field}" in h5:
                    modality_object[field] = h5[f"{modality}/{field}"][start:end]

            self.state[modality] = modality_object

        return self.state


class H5KdDataset(IterableDataset):
    def __init__(self, dataset_path: str, prefix: str, device="cpu", buffer_size: int = 1024,
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

        if num_workers <= len(files):
            files = files[worker_id:: num_workers]
            lengths = lengths[worker_id:: num_workers]

            for file, length in zip(files, lengths):
                yield file, 0, length

            return

        for file, length in zip(files, lengths):
            if length <= 0:
                continue

            num_blocks = (length + self.block_size - 1) // self.block_size
            if num_blocks >= num_workers:
                for block_start in range(0, length, self.block_size):
                    block_id = block_start // self.block_size
                    if block_id % num_workers == worker_id:
                        yield file, block_start, min(block_start + self.block_size, length)

            else:
                step = max(1, length // num_workers)
                start = worker_id * step
                stop = length if worker_id == num_workers - 1 else min(length, (worker_id + 1) * step)

                if start < stop:
                    yield file, start, stop

    def files_for_worker(self, g: torch.Generator):
        perm = torch.randperm(len(self.shard_files), generator=g).tolist()
        files = [self.shard_files[i] for i in perm]
        lengths = [self.shard_lengths[i] for i in perm]

        worker_info = get_worker_info()
        if worker_info is not None:
            files = files[worker_info.id:: worker_info.num_workers]
            lengths = lengths[worker_info.id:: worker_info.num_workers]

        return zip(files, lengths)

    @staticmethod
    def h5_chunk_to_dict(h5_obj: h5py.File | h5py.Group, start: int, stop: int):
        out = {}

        for key, item in h5_obj.items():
            if isinstance(item, h5py.Group):
                out[key] = H5KdDataset.h5_chunk_to_dict(item, start, stop)
            elif isinstance(item, h5py.Dataset):
                arr = item[start:stop]
                if isinstance(arr, np.ndarray) and arr.shape and arr.shape[0] == (stop - start):
                    arr = torch.from_numpy(arr)
                out[key] = arr

        return out

    def iter_shard(self, h5: h5py.File, start: int, stop: int):
        n = int(h5.attrs.get("num_samples", 0)) or int(h5["meta/eid"].shape[0])

        # Clamp
        start, stop = max(0, int(start)), min(int(stop), n)
        if stop <= start:
            self.logger.warning("Start precedes stop? Are you sure?")
            return

        for block_start in range(start, stop, self.block_size):
            block_stop = min(block_start + self.block_size, stop)
            b = block_stop - block_start

            td = TensorDict(self.h5_chunk_to_dict(h5, block_start, block_stop), batch_size=[b])

            for i in range(b):
                yield td[i]

    def __iter__(self):
        # Use a different RNG per worker (and rank), but deterministic across epochs
        worker_info = get_worker_info()
        worker_id = 0 if worker_info is None else worker_info.id

        # This generator behaves the same for all the workers
        global_g = torch.Generator()
        global_g.manual_seed(self.seed + self.epoch)

        # Worker specific
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch + 10 * worker_id)

        warmup: int = min(512, self.buffer_size)

        buffer_growth_rate: int = 64  # How fast I fill the buffer if I am still loading
        ingested_since_yield: int = 0  # Iteration counter to know passed yields

        buffer: list[TensorDict] = []
        for shard_path, start, stop in self.data_for_worker(global_g):
            with h5py.File(str(shard_path), "r") as h5:
                for sample in self.iter_shard(h5, start=start, stop=stop):
                    buffer.append(sample)

                    if len(buffer) < warmup:
                        # You have to at least fill warmup size
                        continue

                    ingested_since_yield += 1
                    if len(buffer) < self.buffer_size and ingested_since_yield < buffer_growth_rate:
                        continue

                    ingested_since_yield = 0
                    # Give a random element
                    yield buffer.pop(torch.randint(len(buffer), (), generator=g).item())

        while buffer:
            # Flush out everything remaining while I can (or else I'd be losing the tail)
            yield buffer.pop(torch.randint(len(buffer), (), generator=g).item())


class H5KdSourceDataset(IterableDataset):
    @staticmethod
    def shard_num_samples(path: Path) -> int:
        with h5py.File(path, "r") as h5:
            return int(h5.attrs.get("num_samples", 0)) or int(h5["meta/eid"].shape[0])

    # todo split on eid? on person? On both?
    # todo mi serve metadata anche quando faccio compressione.
    @staticmethod
    def write_split_manifest(shards_path: str, out_path: str = None, block_size: int = 256, seed: int = 42,
                             val_fraction: float = 0.1, test_fraction: float = 0.15, shuffle_shards: bool = True):
        shards_path: Path = Path(shards_path)

        if out_path is None:
            out_path = shards_path
        out_path: Path = Path(out_path)
        out_path.mkdir(parents=True, exist_ok=True)

        # Read possible shards
        shards = sorted(shards_path.glob("*.h5"))
        assert shards, f"No .h5 in {shards_path}"

        shard_info = [(p.name, H5KdSourceDataset.shard_num_samples(p)) for p in shards]
        total = sum(n for _, n in shard_info)

        n_val, n_test = int(round(val_fraction * total)), int(round(test_fraction * total))
        n_train = total - n_val - n_test

        order = shard_info[:]
        if shuffle_shards:
            rng = random.Random(seed)
            rng.shuffle(order)

        split_names = ["train", "val", "test"]
        splits = {
            "train": {"target": n_train, "rows": [], "count": 0},
            "val": {"target": n_val, "rows": [], "count": 0},
            "test": {"target": n_test, "rows": [], "count": 0},
        }

        cur = 0  # which split we're filling
        for shard_name, n in order:
            start = 0
            while start < n:
                stop = min(start + block_size, n)

                while cur < len(split_names) and splits[split_names[cur]]["count"] >= splits[split_names[cur]][
                    "target"]:
                    cur += 1
                if cur >= len(split_names):
                    # should not happen, but just in case due to rounding
                    cur = len(split_names) - 1
                split_name = split_names[cur]
                splits[split_name]["rows"].append((shard_name, start, stop))
                splits[split_name]["count"] += stop - start

                start = stop

        for split_name in split_names:
            df = pd.DataFrame(splits[split_name]["rows"], columns=["shard", "start", "end"])
            df.to_csv(out_path / f"{split_name}.csv", index=False)
        return {k: splits[k]["count"] for k in splits}, total

    def __init__(self,
                 shards_path: str,
                 manifest_csv_name: str,
                 student_shard_extractor: H5ModalityShardExtractor = H5ModalityShardExtractor.default(),
                 teacher_shard_extractor: H5ModalityShardExtractor = H5ModalityShardExtractor.default(),
                 buffer_size: int = 1024,
                 block_size: int = 256,
                 seed: int = 42,
                 selected_device="cpu",
                 index_cache_name="index.json"):
        super().__init__()
        self.logger = make_logger(self.__class__.__name__)

        self.device = selected_device
        if self.device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # todo use pathlib
        self.shards_root: str = shards_path
        self.shard_paths: list[str] = sorted(glob.glob(os.path.join(shards_path, "*.h5")))
        if not self.shard_paths:
            raise FileNotFoundError(f"No .h5 shards found in: {shards_path}")

        self.buffer_size: int = buffer_size
        if self.buffer_size <= 0:
            raise ValueError("buffer_size must be > 0")

        self.seed: int = seed
        self.epoch: int = 0

        self.block_size: int = block_size
        cache_path = os.path.join(shards_path, index_cache_name)
        self.shard_lengths: list[int] = self.load_lengths(cache_path)
        self.student_shards_extractor: H5ModalityShardExtractor = student_shard_extractor
        self.teacher_shards_extractor: H5ModalityShardExtractor = teacher_shard_extractor

        df = pd.read_csv(shards_path + "/" + manifest_csv_name, dtype={"shard": str, "start": int, "end": int})
        self.entries = list(df.itertuples(index=False, name=None))
        self.length = int(sum(end - start for _, start, end in self.entries))

    def set_epoch(self, epoch: int):
        self.epoch = int(epoch)

    def entries_for_worker(self, g: torch.Generator):
        perm = torch.randperm(len(self.entries), generator=g).tolist()
        entries = [self.entries[i] for i in perm]

        if get_worker_info() is not None:
            entries = entries[get_worker_info().id:: get_worker_info().num_workers]

        return entries

    # todo meh
    def load_lengths(self, cache_path: str):
        # todo use pandas csv
        if os.path.exists(cache_path):
            with open(cache_path, "r") as f:
                obj = json.load(f)
            if obj.get("shards") == self.shard_paths:
                return obj["lengths"]

        lengths = []
        for file_path in self.shard_paths:
            with h5py.File(file_path, "r") as h5:
                # Prefer attr; fallback to dataset length
                n = int(h5.attrs.get("num_samples", 0))
                if n <= 0:
                    # e.g. eid dataset exists
                    n = int(h5["eid"].shape[0]) if "eid" in h5 else 0
                if n <= 0:
                    raise ValueError(f"Could not determine num_samples for shard: {file_path}")
                lengths.append(n)

        with open(cache_path, "w") as f:
            json.dump({"shards": self.shard_paths, "lengths": lengths}, f)

        return lengths

    def get_shards_for_worker(self, g: torch.Generator):
        permutation = torch.randperm(len(self.shard_paths), generator=g).tolist()
        shard_paths = [self.shard_paths[i] for i in permutation]

        info = get_worker_info()  # Assign to each worker exclusive shards. It is best if the workers have more than one shard for local shuffling.
        if info is not None:
            shard_paths = shard_paths[info.id:: info.num_workers]

        return shard_paths

    def iter_shard(self, h5: h5py.File, start: int, stop: int) -> Iterator[dict[str, dict]]:
        n = int(h5.attrs.get("num_samples", 0)) or int(h5["meta/eid"].shape[0])

        # Clamp
        start = max(0, int(start))
        stop = min(int(stop), n)

        if stop <= start:
            return

        for s in range(start, stop, self.block_size):
            e = min(s + self.block_size, stop)
            # Read meta in one go
            eid_list = h5["meta/eid"][s:e]
            idx_list = h5["meta/index"][s:e]

            self.student_shards_extractor.make(h5["student"], s, e)
            self.teacher_shards_extractor.make(h5["teacher"], s, e)

            for i in range(e - s):
                yield {
                    "eid": eid_list[i].decode() if hasattr(eid_list[i], "decode") else eid_list[i],
                    "idx": idx_list[i],
                    "student": self.student_shards_extractor.take(i),
                    "teacher": self.teacher_shards_extractor.take(i)
                }

    def __iter__(self):
        # Use a different RNG per worker (and rank), but deterministic across epochs
        worker_info = get_worker_info()
        worker_id = 0 if worker_info is None else worker_info.id

        # This generator behaves the same for all the workers
        global_g = torch.Generator()
        global_g.manual_seed(self.seed + self.epoch)
        # Worker specific
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch + 10 * worker_id)

        warmup: int = min(512, self.buffer_size)
        buffer_growth_rate: int = 64  # How fast I fill the buffer if I am still loading
        ingested_since_yield: int = 0  # Iteration counter to know passed yields

        buffer: list[dict[str, Any]] = []
        for shard_name, start, stop in self.entries_for_worker(global_g):
            with h5py.File(str(Path(self.shards_root) / shard_name), "r") as h5:
                for sample in self.iter_shard(h5, start=start, stop=stop):
                    buffer.append(sample)

                    if len(buffer) < warmup:
                        continue  # You have to at least fill warmup size

                    ingested_since_yield += 1
                    if len(buffer) < self.buffer_size:
                        if ingested_since_yield < buffer_growth_rate:
                            continue
                        ingested_since_yield = 0
                    else:
                        # This if for the 1-in / 1-out
                        ingested_since_yield = 0

                    # Give a random element
                    yield buffer.pop(torch.randint(len(buffer), (), generator=g).item())

        while buffer:
            yield buffer.pop(torch.randint(len(buffer), (), generator=g).item())

    def __len__(self) -> int:
        return self.length


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


class RoundRobinMultiDataset(IterableDataset):
    def __init__(self, datasets: list[IterableDataset], weights, seed: int):
        super().__init__()
        self.logger = make_logger(self.__class__.__name__)
        self.datasets: list[IterableDataset] = datasets

        w = torch.tensor(weights, dtype=torch.float)
        self.weights = w

        self.epoch = 0
        self.seed = seed

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
            try:
                yield next(iters[k])
                dead[k] = 0

            except StopIteration:
                dead[k] += 1
                if dead[k] > 5:
                    raise RuntimeError(f"Dataset {k} keeps stopping; is it empty?")
                # Restart that dataset (cycle)
                iters[k] = iter(self.datasets[k])


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


@dataclasses.dataclass
class QueueState:
    perm: torch.Tensor  # permutation of [0..length-1]
    ptr: int  # next position to read


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

# Create WebDataset shards. But now we have to change structure so that:
# - Each shard contains samples from across the whole dataset (Local ds so AMIGOS only AMIGOS) randomly picked
# - For performance reasons having both teacher and student inputs -> One record has to contain both teacher student value and no longer 2 different partitions.
# - On load of a shard for first time in epoch shuffle it. Take the first B samples available (exhaustion map  to track what not to take)
import dataclasses
import logging
import os
import random
from collections import defaultdict, deque
from pathlib import Path
from typing import Optional, Any

import h5py
import numpy as np
import pandas as pd
import tensordict
import torch
from h5py import Dataset
from tensordict import TensorDict
from tensordict._tensorcollection import TensorCollection
from tqdm import tqdm

from main.core_data.media.metadata.metadata import Metadata
from main.utils.logging import make_logger


def materialize(x):
    if hasattr(x, "to_tensor"):  # MemoryMappedTensor
        return x.to_tensor()
    if hasattr(x, "clone"):
        return x.clone()
    return x


def to_numpy(x: torch.Tensor):
    return materialize(x).detach().cpu().numpy()


@dataclasses.dataclass()
class MetaInformation:
    eid_ds: h5py.Dataset
    idx_ds: h5py.Dataset
    ds_id: h5py.Dataset
    experiment: h5py.Dataset
    interval: h5py.Dataset


class KdDataSharder:
    def __init__(self,
                 student_spec_path: str,
                 teacher_spec_path: str,
                 output_path: str,
                 shard_size_gb=4,
                 compression=None,
                 val_participants: int = 0,
                 test_participants: int = 0,
                 uid_store_path: str = None,
                 min_chunk_size: int = 1,
                 max_chunk_size: int = 4096
                 ):
        self.logger = make_logger(self.__class__.__name__)

        self.student_path = Path(student_spec_path).parent
        self.student_df = pd.read_csv(student_spec_path)

        self.teacher_path = Path(teacher_spec_path).parent
        self.teacher_df = pd.read_csv(teacher_spec_path)

        tmap = {}
        for row in self.teacher_df.itertuples(index=False):
            # Expects columns: eid, index, sharded_eid, sharded_index
            tmap[(str(row.eid), row.index)] = (str(row.sharded_eid), int(row.sharded_index))

        self.teacher_map = tmap

        self.output_path: Path = Path(output_path)
        Path(self.output_path).mkdir(parents=True, exist_ok=True)
        self.shard_size_bytes = int(shard_size_gb * (1024 ** 3))

        self.student_current_td: Optional[TensorDict] = None
        self.current_student_name: Optional[str] = None
        self.teacher_current_td: Optional[TensorDict] = None
        self.current_teacher_name: Optional[str] = None

        self.min_chunk_size: int = min_chunk_size
        self.max_chunk_size: int = max_chunk_size
        # Commonly accepted heuristic on size of chunks for I/O and CPU tradeoff
        self.target_chunk_mb = 8

        self.compression = compression
        self.uid_store_path: str = uid_store_path
        self.val_participants: float = val_participants
        self.test_participants: float = test_participants

        # todo build hjold out sets (test/val) by removing people. + some experiment entirely (while not getting too big) tkae 25% of people?

    def load_student_shard(self, shard_name: str):
        if shard_name != self.current_student_name:
            self.student_current_td = tensordict.load_memmap(self.student_path / shard_name)
            self.current_student_name = shard_name
        return self.student_current_td

    def load_teacher_shard(self, shard_name: str):
        if shard_name != self.current_teacher_name:
            self.teacher_current_td = tensordict.load_memmap(self.teacher_path / shard_name)
            self.current_teacher_name = shard_name
        return self.teacher_current_td

    @staticmethod
    def ensure_meta_appendable(h5: h5py.File) -> MetaInformation:
        exp_key = "meta/experiment"
        int_key = "meta/interval"
        if "meta/eid" in h5 and "meta/index" in h5 and "meta/dataset_id" in h5 and exp_key in h5 and int_key in h5:
            return MetaInformation(h5["meta/eid"], h5["meta/index"], h5["meta/dataset_id"], h5[exp_key], h5[int_key])

        str_datatype = h5py.string_dtype(encoding="utf-8")
        return MetaInformation(
            eid_ds=h5.create_dataset("meta/eid", shape=(0,), maxshape=(None,), dtype=str_datatype, chunks=(1024,)),
            idx_ds=h5.create_dataset("meta/index", shape=(0,), maxshape=(None,), dtype=np.int64, chunks=(1024,)),
            ds_id=h5.create_dataset("meta/dataset_id", shape=(0,), maxshape=(None,), dtype=np.int64, chunks=(1024,)),
            experiment=h5.create_dataset(exp_key, shape=(0,), maxshape=(None,), dtype=str_datatype, chunks=(1024,)),
            interval=h5.create_dataset(int_key, shape=(0, 2), maxshape=(None, 2), dtype=np.float16, chunks=(1024, 2))
        )

    def open_new_shard(self, h5: Optional[h5py.File], shard_id: int, shard_size: int, shard_name: str):
        self._close_danling_shard(h5, shard_size)
        shard_name = f"{shard_name}_{shard_id:03d}.h5"
        output_path = self.output_path / shard_name

        h5 = h5py.File(output_path, "w")
        meta = self.ensure_meta_appendable(h5)
        # Returns the h5, path, new shard_id, meta_ds collection and i_in_shard
        return h5, output_path, shard_id + 1, meta, 0

    def extract_rows_by_eid(self) -> tuple[defaultdict[Any, list], defaultdict[Any, list], defaultdict[Any, list]]:
        # THis extracts the rows based on persons for test/val (We build based on unseen people)
        missing_val, missing_test = self.val_participants, self.test_participants
        train_eid, val_eid, test_eid = defaultdict(list), defaultdict(list), defaultdict(list)

        for record in self.student_df.itertuples(index=False):
            # todo check person id not eid. Posso usare uid stored? No perche non ha eid quindi useless
            #       in preprocessing dovrei estrarre anche persona su csv metadata. A questo punto posso anche shufflare quali prendere.
            # EID is combination of person + experiment
            if missing_val > 0:
                val_eid[str(record.eid)].append(record)
                missing_val -= 1
            elif missing_test > 0:
                test_eid[str(record.eid)].append(record)
                missing_test -= 1
            else:
                train_eid[str(record.eid)].append(record)

        return train_eid, val_eid, test_eid

    @staticmethod
    def _close_danling_shard(h5: Optional[h5py.File], i: int):
        if h5 is not None:
            h5.attrs["num_samples"] = i

            h5.flush()
            h5.close()

    def build(self):
        train_by_eid, val_by_eid, test_by_eid = self.extract_rows_by_eid()

        # Seed for reproducibility
        rng = random.Random(123)
        for eid, bucket in train_by_eid.items():
            rng.shuffle(bucket)

        self.elaborate_for_modality("train", train_by_eid)
        if self.val_participants > 0:
            self.elaborate_for_modality("val", val_by_eid)
        if self.test_participants > 0:
            self.elaborate_for_modality("test", test_by_eid)

    def elaborate_for_modality(self, modality: str, rows_by_eid: defaultdict[Any, list]):
        self.logger.info(f"Working for modality: {modality}")
        active_eid_collection = deque(rows_by_eid.keys())
        pos = {eid: 0 for eid in rows_by_eid.keys()}

        total_written: int = 0

        h5: h5py.File
        shard_id: int  # We start at 0 of course
        h5, current_path, shard_id, meta_ds_collection, i_in_shard = self.open_new_shard(
            h5=None, shard_id=0, shard_size=0, shard_name=modality
        )

        last_read_filesize: int = 0
        pbar = tqdm(total=self.shard_size_bytes, desc="Resharding", unit="B", unit_scale=True, unit_divisor=1024)

        while active_eid_collection:
            curr_i = self.consume_collection(
                h5=h5,
                eid_collection=active_eid_collection,
                pos_track=pos,
                rows_by_eid=rows_by_eid,
                meta_ds_collection=meta_ds_collection,
                i_in_shard=i_in_shard
            )

            if curr_i != i_in_shard:
                total_written += 1

            i_in_shard = curr_i
            if total_written % 256 == 0:
                h5.flush()

                pbar.update(os.path.getsize(current_path) - last_read_filesize)
                pbar.set_postfix(samples_done=total_written)
                last_read_filesize = os.path.getsize(current_path)

                if os.path.getsize(current_path) >= self.shard_size_bytes:
                    pbar.close()
                    pbar = tqdm(total=self.shard_size_bytes, desc=f"Shard {shard_id}",
                                unit="B", unit_scale=True, unit_divisor=1024)

                    # Reset the read filesize
                    last_read_filesize = 0
                    logging.info(f"Creating new shard #{shard_id}")
                    h5, current_path, shard_id, meta_ds_collection, i_in_shard = self.open_new_shard(
                        h5=h5, shard_id=shard_id, shard_size=i_in_shard, shard_name="train"
                    )

        self._close_danling_shard(h5, i_in_shard)

    def ensure_tensor_ds_appendable(self, h5: h5py.File, path: str, sample: np.ndarray):
        if path in h5:
            return h5[path]

        chunk0 = self.choose_chunk0(sample)
        has_shape = sample.shape != ()
        return h5.create_dataset(
            path,
            dtype=sample.dtype,
            shape=(0,) + sample.shape if has_shape else (0,),
            maxshape=(None,) + sample.shape if has_shape else (None,),
            chunks=(chunk0,) + sample.shape if has_shape else (chunk0,),
            compression=self.compression,
            shuffle=True,  # Recommended if you use gzip
        )

    def choose_chunk0(self, sample: np.ndarray):
        bytes_per_sample = sample.nbytes
        if bytes_per_sample == 0:
            return 1

        target_bytes = self.target_chunk_mb * 1024 * 1024
        chunk0 = max(self.min_chunk_size, target_bytes // bytes_per_sample)
        return int(max(self.min_chunk_size, min(chunk0, self.max_chunk_size)))

    def append_records(self, h5: h5py.File, td: TensorDict, base: str, i: int, skip_modalities: list[str] = ("meta",)):
        for modality, container in td.items():
            if modality in skip_modalities:
                continue
            for key, value in container.items():
                arr = to_numpy(value)
                ds_path: str = f"{base}/{modality}/{key}"

                ds = self.ensure_tensor_ds_appendable(h5, ds_path, arr)
                if ds.shape[0] <= i:
                    ds.resize((i + 1,) + ds.shape[1:])

                ds[i, ...] = arr

    def consume_collection(self, h5: h5py.File, eid_collection: deque[Any], pos_track: dict, rows_by_eid: dict,
                           meta_ds_collection: MetaInformation, i_in_shard: int):
        eid = eid_collection.popleft()
        bucket = rows_by_eid[eid]
        if pos_track[eid] >= len(bucket):
            self.logger.info(f"eid:{eid} exhausted")
            return i_in_shard

        record = bucket[pos_track[eid]]
        pos_track[eid] += 1  # Increase the positions used tracker
        eid_collection.append(eid)  # Still active (unless it becomes exhausted later)

        idx = int(record.index)

        teacher_location = self.teacher_map.get((eid, idx))
        if teacher_location is None:
            self.logger.warn(f"Missing record for tuple ({eid}, {idx}) in teacher!")
            return i_in_shard  # No existing row so we stop for this sample

        student_shard, student_i = str(record.sharded_eid), int(record.sharded_index)
        teacher_shard, teacher_i = teacher_location

        student_td = self.load_student_shard(student_shard)
        teacher_td = self.load_teacher_shard(teacher_shard)

        if not (teacher_td["meta"][teacher_i] == student_td["meta"][student_i]).all():
            raise ValueError("Sample mismatch!")

        if meta_ds_collection.eid_ds.shape[0] <= i_in_shard:
            meta_ds_collection.eid_ds.resize((i_in_shard + 1,))
            meta_ds_collection.idx_ds.resize((i_in_shard + 1,))
            meta_ds_collection.ds_id.resize((i_in_shard + 1,))
            meta_ds_collection.experiment.resize((i_in_shard + 1,))
            meta_ds_collection.interval.resize((i_in_shard + 1, 2))

        meta_ds_collection.ds_id[i_in_shard] = student_td["meta"]["dataset_id"][student_i]
        meta_ds_collection.eid_ds[i_in_shard] = eid
        meta_ds_collection.idx_ds[i_in_shard] = idx
        meta_ds_collection.experiment[i_in_shard] = student_td["meta"]["experiment"][student_i]
        meta_ds_collection.interval[i_in_shard, :] = to_numpy(student_td["meta"]["interval"][student_i])

        self.append_records(h5, student_td[student_i], "student", i_in_shard)
        self.append_records(h5, teacher_td[teacher_i], "teacher", i_in_shard)

        # Add one element
        return i_in_shard + 1


# Hold out 1 experiment (or at most ~10–20% of experiments) across all training participants
# your test percentage should be defined in participants (or participant×experiment groups), not in windows
# TODO holdout AMIGOS 6 EAV 6 DEAP 5 (15%)


class FusedDataSharder:
    def __init__(
            self,
            spec_path: str, output_path: str,
            val_participants: int = 0, test_participants: int = 0,
            min_chunk_size: int = 1, max_chunk_size: int = 4096, shard_size_gb: int = 16, compression=None,
            chunk0: int = 256
    ):
        self.logger = make_logger(self.__class__.__name__)
        self.df = pd.read_csv(spec_path)

        self.data_path: Path = Path(spec_path).parent
        self.output_path: Path = Path(output_path)
        Path(self.output_path).mkdir(parents=True, exist_ok=True)
        self.shard_size_bytes = int(shard_size_gb * (1024 ** 3))

        self.current_td: Optional[TensorDict] = None
        self.current_td_name: Optional[str] = None

        self.min_chunk_size: int = min_chunk_size
        self.max_chunk_size: int = max_chunk_size

        # Commonly accepted heuristic on size of chunks for I/O and CPU tradeoff
        self.target_chunk_mb = 8
        self.compression = compression

        # For creating val/test sets based on people
        self.val_participants: float = val_participants
        self.test_participants: float = test_participants
        # Optimization choice
        self.capacity_by_path: dict[str, int] = {}
        self.grow_by: int = 256  # tune: 256/1024/4096

        self.chunk0 = chunk0

    def load_td(self, td_name: str):
        if td_name != self.current_td_name:
            self.current_td = tensordict.load_memmap(self.data_path / str(td_name))
            self.current_td_name = td_name

        return self.current_td

    @staticmethod
    def trim(h5: h5py.File, n: int):
        def _visit(name, obj):
            if not isinstance(obj, h5py.Dataset) or obj.ndim < 1:
                return

            if obj.maxshape is None or obj.maxshape[0] is None:
                if obj.shape[0] > n:
                    obj.resize((n,) + obj.shape[1:])

        h5.visititems(_visit)

    def close_dandling_shard(self, h5: Optional[h5py.File], i: int) -> None:
        if h5 is not None:
            self.trim(h5, i)
            h5.attrs["num_samples"] = i
            h5.close()

    def open_new_shard(self, h5: Optional[h5py.File], shard_id: int, shard_size: int, shard_name: str):
        self.close_dandling_shard(h5, shard_size)
        shard_name = f"{shard_name}_{shard_id:03d}.h5"
        output_path = self.output_path / shard_name
        # Returns the h5, path, new shard_id and i_in_shard
        return h5py.File(output_path, "w"), output_path, shard_id + 1, 0

    def extract_rows(self):
        # Considers train/test/val for eid partitioning
        persons = self.df["person_id"].unique().tolist()

        if self.val_participants + self.test_participants > len(persons):
            raise ValueError("val_participants + test_participants > number of unique persons")

        rng = random.Random(42)
        rng.shuffle(persons)

        val_persons = set(persons[0:self.val_participants])
        test_persons = set(persons[self.val_participants:self.val_participants + self.test_participants])

        train_eid, val_eid, test_eid = defaultdict(list), defaultdict(list), defaultdict(list)

        for record in self.df.itertuples(index=False):
            if record.person_id in val_persons:
                val_eid[record.eid].append(record)

            elif record.person_id in test_persons:
                test_eid[record.eid].append(record)

            else:
                train_eid[record.eid].append(record)

        return train_eid, val_eid, test_eid

    def run(self):
        train_samples, validation_samples, test_samples = self.extract_rows()

        rng = random.Random(123)
        for eid, bucket in train_samples.items():
            rng.shuffle(bucket)

        self.elaborate_modality("train", train_samples)
        if self.val_participants > 0:
            self.elaborate_modality("val", validation_samples)
        if self.test_participants > 0:
            self.elaborate_modality("test", test_samples)

    def choose_chunk0(self, sample: np.ndarray):
        bytes_per_sample = sample.nbytes
        if bytes_per_sample == 0:
            return 1

        target_bytes = self.target_chunk_mb * 1024 * 1024
        chunk0 = max(self.min_chunk_size, target_bytes // bytes_per_sample)
        return int(max(self.min_chunk_size, min(chunk0, self.max_chunk_size)))

    def ensure_tensor_ds_appendable(self, h5: h5py.File, path: str, sample: np.ndarray) -> Dataset:
        # todo make meta also correctly (strings?)
        if path in h5:
            return h5[path]

        # chunk0 = self.choose_chunk0(sample)
        is_str = sample.dtype.kind in ("U", "S") or sample.dtype == object
        has_shape = sample.shape != ()

        return h5.create_dataset(
            name=path,
            shape=(0,) + sample.shape if has_shape else (0,),
            maxshape=(None,) + sample.shape if has_shape else (None,),
            dtype=h5py.string_dtype("utf-8") if is_str else sample.dtype,
            chunks=(self.chunk0,) + sample.shape if has_shape else (self.chunk0,),
            compression=self.compression if not is_str else None,
        )

    def do_append(self, h5: h5py.File, wrapper: TensorCollection, prefix: str, count: int):
        for key, value in wrapper.items():
            arr = to_numpy(value)
            if (isinstance(arr, dict) and arr == {}) or arr.dtype.kind in ("U", "S") or arr.dtype == object:
                continue

            ds_path = f"{prefix}/{key}"
            ds = self.ensure_tensor_ds_appendable(h5, ds_path, arr)

            cap = self.capacity_by_path.get(ds_path, ds.shape[0])
            if count >= cap:
                new_cap = max(count + 1, cap + self.grow_by)
                ds.resize((new_cap,) + ds.shape[1:])

                self.capacity_by_path[ds_path] = new_cap

            if arr.shape == ():
                ds[count] = arr.item()
            else:
                ds[count, ...] = arr

    def append_records(self, h5: h5py.File, td: TensorDict, count: int):
        # This is where the class is a little ugly.
        for wrapper_key, wrapper in td.items():
            modality: TensorCollection  # We have dicts of dicts

            if wrapper_key == Metadata.modality_code():
                self.do_append(h5, wrapper, wrapper_key, count)
                continue  # Metadata handler

            for modality_key, modality in wrapper.items():
                self.do_append(h5, modality, f"{wrapper_key}/{modality_key}", count)

    def consume(self, h5: h5py.File, eid: str, pos_track: dict, rows_by_eid: dict, count: int):
        bucket = rows_by_eid[eid]

        if pos_track[eid] >= len(bucket):
            self.logger.info(f"eid:{eid} exhausted")
            return count

        record = bucket[pos_track[eid]]
        pos_track[eid] += 1  # Increase the positions used tracker
        idx = record.index

        td = self.load_td(eid)
        self.append_records(h5, td[idx], count)
        return count + 1

    def elaborate_modality(self, modality: str, rows_by_eid: defaultdict[Any, list]):
        self.logger.info(f"Working for modality: {modality}")
        # Reset
        self.capacity_by_path = {}
        active_eid_collection = deque(rows_by_eid)

        pos = {eid: 0 for eid in rows_by_eid.keys()}

        last_read_filesize: int = 0
        h5, current_path, shard_id, count = self.open_new_shard(
            h5=None, shard_id=0, shard_size=0, shard_name=modality
        )
        total_written = 0
        pbar = tqdm(total=self.shard_size_bytes, desc="Resharding", unit="B", unit_scale=True, unit_divisor=1024)
        while active_eid_collection:
            eid = active_eid_collection.popleft()
            current = self.consume(h5=h5, eid=eid, pos_track=pos, rows_by_eid=rows_by_eid, count=count, )

            if current != count:
                active_eid_collection.append(eid)
                total_written += 1

            count = current
            if total_written % 256 == 0:
                h5.flush()

                pbar.update(os.path.getsize(current_path) - last_read_filesize)
                pbar.set_postfix(samples_done=total_written)
                last_read_filesize = os.path.getsize(current_path)

                if os.path.getsize(current_path) >= self.shard_size_bytes:
                    pbar.close()
                    pbar = tqdm(total=self.shard_size_bytes, desc=f"Shard {shard_id}",
                                unit="B", unit_scale=True, unit_divisor=1024)

                    # Reset the read filesize
                    last_read_filesize = 0
                    logging.info(f"Creating new shard #{shard_id}")
                    h5, current_path, shard_id, count = self.open_new_shard(
                        h5=h5, shard_id=shard_id, shard_size=count, shard_name=modality
                    )
                    # Reset capacities as we start a new file
                    self.capacity_by_path = {}

        self.close_dandling_shard(h5, count)

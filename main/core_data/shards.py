# Create WebDataset shards. But now we have to change structure so that:
# - Each shard contains samples from across the whole dataset (Local ds so AMIGOS only AMIGOS) randomly picked
# - For performance reasons having both teacher and student inputs -> One record has to contain both teacher student value and no longer 2 different partitions.
# - On load of a shard for first time in epoch shuffle it. Take the first B samples available (exhaustion map  to track what not to take)
import logging
import os
import random
from collections import defaultdict, deque
from pathlib import Path
from typing import Optional

import h5py
import numpy as np
import pandas as pd
import tensordict
import torch
from tensordict import TensorDict
from tqdm import tqdm

from main.utils.logging import make_logger


def materialize(x):
    if hasattr(x, "to_tensor"):  # MemoryMappedTensor
        return x.to_tensor()
    if hasattr(x, "clone"):
        return x.clone()
    return x


def to_numpy(x: torch.Tensor):
    return materialize(x).detach().cpu().numpy()


class KdDataSharder:
    def __init__(self,
                 student_spec_path: str,
                 teacher_spec_path: str,
                 output_path: str,
                 shard_size_gb=4,
                 compression=None,
                 val_participants: int = 0,
                 test_participants: int = 0,
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



# Hold out 1 experiment (or at most ~10–20% of experiments) across all training participants
# your test percentage should be defined in participants (or participant×experiment groups), not in windows
# TODO holdout AMIGOS 6 EAV 6 DEAP 5 (15%)
class ReSharder:
    def __init__(self, student_spec_path: str, teacher_spec_path: str, output_path: str,
                 shard_size_gb=4, compression=None, min_chunk_size: int = 1, max_chunk_size: int = 4096):
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
        self.prefix: str = "ds"

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

    def count_samples(self):
        count: int = 0
        for record in self.student_df.itertuples(index=False):
            # Only if the teacher equivalent exists
            if (str(record.eid), int(record.index)) in self.teacher_map:
                count += 1

        return count

    @staticmethod
    def ensure_meta_appendable(h5: h5py.File):
        if "meta/eid" in h5 and "meta/index" in h5:
            return h5["meta/eid"], h5["meta/index"]

        str_dt = h5py.string_dtype(encoding="utf-8")
        eid_ds = h5.create_dataset("meta/eid", shape=(0,), maxshape=(None,), dtype=str_dt, chunks=(1024,))
        idx_ds = h5.create_dataset("meta/index", shape=(0,), maxshape=(None,), dtype=np.int64, chunks=(1024,))
        ds_id = h5.create_dataset("meta/dataset_id", shape=(0,), maxshape=(None,), dtype=np.int64, chunks=(1024,))
        experiment = h5.create_dataset("meta/experiment", shape=(0,), maxshape=(None,), dtype=str_dt, chunks=(1024,))
        interval = h5.create_dataset("meta/interval", shape=(0, 2), maxshape=(None, 2), dtype=np.float64,
                                     chunks=(1024, 2))

        return eid_ds, idx_ds, ds_id, experiment, interval

    def open_new_shard(self, h5: h5py.File, shard_id: int, shard_size: int):
        if h5 is not None:
            h5.attrs["num_samples"] = shard_size
            # Close the current shard
            h5.flush()
            h5.close()

        shard_name = f"{self.prefix}_{shard_id:03d}.h5"
        output_path = self.output_path / shard_name
        h5 = h5py.File(output_path, "w")

        eid_ds, idx_ds, ds_id, experiment, interval = self.ensure_meta_appendable(h5)
        i_in_shard: int = 0
        shard_id += 1

        return h5, output_path, shard_id, shard_name, eid_ds, idx_ds, ds_id, experiment, interval, i_in_shard

    def choose_chunk0(self, sample: np.ndarray):
        bytes_per_sample = sample.nbytes
        if bytes_per_sample == 0:
            return 1

        target_bytes = self.target_chunk_mb * 1024 * 1024
        chunk0 = max(self.min_chunk_size, target_bytes // bytes_per_sample)
        return int(max(self.min_chunk_size, min(chunk0, self.max_chunk_size)))

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

    def append_records(self, h5: h5py.File, td: TensorDict, base: str, i: int,
                       ignore_modalities: list[str] = ("meta",)):
        for modality, container in td.items():
            if modality in ignore_modalities:
                continue
            for key, value in container.items():
                arr = to_numpy(value)
                ds_path: str = f"{base}/{modality}/{key}"

                ds = self.ensure_tensor_ds_appendable(h5, ds_path, arr)
                if ds.shape[0] <= i:
                    ds.resize((i + 1,) + ds.shape[1:])

                ds[i, ...] = arr

    def build(self):
        # Take on from each experiment in order (Round Robin)
        rows_by_eid = defaultdict(list)
        # Each sample with same eid is aggregated for later usage.
        for record in self.student_df.itertuples(index=False):
            rows_by_eid[str(record.eid)].append(record)

        # Seed for reproducibility
        rng = random.Random(123)
        for eid, bucket in rows_by_eid.items():
            rng.shuffle(bucket)

        active_eid_collection = deque(rows_by_eid.keys())
        # Pos tracks how many times one element was used
        pos = {eid: 0 for eid in rows_by_eid.keys()}

        shard_id: int = 0
        h5: Optional[h5py.File] = None
        total_written: int = 0

        h5, current_path, shard_id, current_name, eid_ds, idx_ds, ds_id, experiment_ds, interval_ds, i_in_shard = self.open_new_shard(
            h5=h5, shard_id=shard_id, shard_size=0
        )

        last_read_filesize = 0
        pbar = tqdm(total=self.shard_size_bytes, desc="Resharding", unit="B", unit_scale=True, unit_divisor=1024)
        while active_eid_collection:
            eid = active_eid_collection.popleft()
            bucket = rows_by_eid[eid]
            # If all were placed we just stop using the current eid
            if pos[eid] >= len(bucket):
                self.logger.info(f"eid:{eid} exhausted")
                continue

            record = bucket[pos[eid]]
            pos[eid] += 1
            active_eid_collection.append(eid)  # Still active (unless it becomes exhausted later)

            idx = int(record.index)

            # This mapping should be solid but no harm in checking TODO
            teacher_location = self.teacher_map.get((eid, idx))
            if teacher_location is None:
                self.logger.warn(f"Missing record for tuple ({eid}, {idx}) in teacher!")
                continue  # No existing row

            student_shard, student_i = str(record.sharded_eid), int(record.sharded_index)
            teacher_shard, teacher_i = teacher_location

            student_td = self.load_student_shard(student_shard)
            meta_student = student_td["meta"]

            teacher_td = self.load_teacher_shard(teacher_shard)
            meta_teacher = teacher_td["meta"]

            # todo check meta student and teacher eq
            if not (meta_teacher[teacher_i] == meta_student[student_i]).all():
                raise ValueError("Sample mismatch!")

            student_record = student_td[student_i]
            teacher_record = teacher_td[teacher_i]

            if eid_ds.shape[0] <= i_in_shard:
                eid_ds.resize((i_in_shard + 1,))
                idx_ds.resize((i_in_shard + 1,))
                ds_id.resize((i_in_shard + 1,))
                experiment_ds.resize((i_in_shard + 1,))
                interval_ds.resize((i_in_shard + 1, 2))

            ds_id[i_in_shard] = meta_student["dataset_id"][student_i]
            eid_ds[i_in_shard] = eid
            idx_ds[i_in_shard] = idx
            experiment_ds[i_in_shard] = meta_student["experiment"][student_i]
            interval_ds[i_in_shard, :] = to_numpy(meta_student["interval"][student_i])

            self.append_records(h5, student_record, "student", i_in_shard)
            self.append_records(h5, teacher_record, "teacher", i_in_shard)

            i_in_shard += 1
            total_written += 1
            if (total_written % 256) == 0:  # don’t stat every sample
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
                    h5, current_path, shard_id, current_name, eid_ds, idx_ds, ds_id, experiment_ds, interval_ds, i_in_shard = self.open_new_shard(
                        h5=h5, shard_id=shard_id, shard_size=i_in_shard
                    )
                self.logger.info(f"Written a total ")

        if h5 is not None:
            h5.attrs["num_samples"] = i_in_shard
            h5.flush()
            h5.close()

        return self.output_path

# Create WebDataset shards. But now we have to change structure so that:
# - Each shard contains samples from across the whole dataset (Local ds so AMIGOS only AMIGOS) randomly picked
# - For performance reasons having both teacher and student inputs -> One record has to contain both teacher student value and no longer 2 different partitions.
# - On load of a shard for first time in epoch shuffle it. Take the first B samples available (exhaustion map  to track what not to take)
import os
from collections import defaultdict, deque
from pathlib import Path
from typing import Optional

import h5py
import numpy as np
import pandas as pd
import tensordict
import torch
from tensordict import TensorDict


def materialize(x):
    if hasattr(x, "to_tensor"):  # MemoryMappedTensor
        return x.to_tensor()
    if hasattr(x, "clone"):
        return x.clone()
    return x


def to_numpy(x: torch.Tensor):
    return materialize(x).detach().cpu().numpy()


class ReSharder:
    def __init__(self, student_spec_path: str, teacher_spec_path: str, output_path: str, shard_size_gb=4):
        self.student_path = Path(student_spec_path).parent
        self.student_df = pd.read_csv(student_spec_path)

        self.teacher_path = Path(teacher_spec_path).parent
        self.teacher_df = pd.read_csv(teacher_spec_path)

        tmap = {}
        for row in self.teacher_df.itertuples(index=False):
            # expects columns: eid, index, sharded_eid, sharded_idx
            tmap[(str(row.eid), row.index)] = (str(row.sharded_eid), int(row.sharded_index))
        self.teacher_map = tmap

        self.output_path: Path = Path(output_path)
        Path(self.output_path).mkdir(parents=True, exist_ok=True)
        self.shard_size_bytes = int(shard_size_gb * (1024 ** 3))

        self.student_current_td: Optional[TensorDict] = None
        self.current_student_name: Optional[str] = None
        self.teacher_current_td: Optional[TensorDict] = None
        self.current_teacher_name: Optional[str] = None

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

        return eid_ds, idx_ds

    def open_new_shard(self, h5: h5py.File, shard_id: int, shard_size: int):
        if h5 is not None:
            h5.attrs["num_samples"] = shard_size
            # Close the current shard
            h5.flush()
            h5.close()

        shard_name = f"{self.prefix}_{shard_id:03d}.h5"
        output_path = self.output_path / shard_name
        h5 = h5py.File(output_path, "w")

        eid_ds, idx_ds = self.ensure_meta_appendable(h5)
        i_in_shard: int = 0
        shard_id += 1

        return h5, output_path, shard_id, shard_name, eid_ds, idx_ds, i_in_shard

    @staticmethod
    def ensure_tensor_ds_appendable(h5: h5py.File, path: str, sample: np.ndarray, chunk0: int = 64, compression=None):
        if path in h5:
            return h5[path]

        has_shape = sample.shape != ()
        return h5.create_dataset(
            path,
            dtype=sample.dtype,
            shape=(0,) + sample.shape if has_shape else (0,),
            maxshape=(None,) + sample.shape if has_shape else (None,),
            chunks=(chunk0,) + sample.shape if has_shape else (chunk0,),
            compression=compression
        )

    def append_records(self, h5: h5py.File, td: TensorDict, base: str, i: int, chunk0: int = 64, compression=None):
        for modality, container in td.items():
            for key, value in container.items():
                arr = to_numpy(value)
                ds_path: str = f"{base}/{modality}/{key}"

                ds = self.ensure_tensor_ds_appendable(h5, ds_path, arr, chunk0=chunk0, compression=compression)

                if ds.shape[0] <= i:
                    ds.resize((i + 1,) + ds.shape[1:])

                ds[i, ...] = arr

    def build(self, compression=None, chunk0: int = 64):
        # Take on from each experiment in order (Round Robin)
        rows_by_eid = defaultdict(list)
        for record in self.student_df.itertuples(index=False):
            rows_by_eid[str(record.eid)].append(record)

        active_eids = deque(rows_by_eid.keys())
        pos = {eid: 0 for eid in rows_by_eid.keys()}

        shard_id: int = 0
        h5: Optional[h5py.File] = None
        total_written: int = 0

        h5, current_path, shard_id, current_name, eid_ds, idx_ds, i_in_shard = self.open_new_shard(
            h5=h5, shard_id=shard_id, shard_size=0
        )

        # TODO csv sbalgiato? nomi migliori
        while active_eids:
            eid = active_eids.popleft()
            p = pos[eid]
            bucket = rows_by_eid[eid]

            if p >= len(bucket):
                continue  # this eid exhausted

            record = bucket[p]
            pos[eid] = p + 1
            active_eids.append(eid)  # still active (unless it becomes exhausted later)

            idx = int(record.index)

            # This mapping should be solid but no harm in checking TODO
            teacher_location = self.teacher_map.get((eid, idx))
            if teacher_location is None:
                continue  # No existing row

            student_shard, student_i = str(record.sharded_eid), int(record.sharded_index)
            teacher_shard, teacher_i = teacher_location

            student_td = self.load_student_shard(student_shard)
            student_td["meta"].pop("experiment", None)  # TODO arricchiro il csv

            teacher_td = self.load_teacher_shard(teacher_shard)
            teacher_td["meta"].pop("experiment", None)

            student_record = student_td[student_i]
            teacher_record = teacher_td[teacher_i]

            if eid_ds.shape[0] <= i_in_shard:
                eid_ds.resize((i_in_shard + 1,))
                idx_ds.resize((i_in_shard + 1,))

            eid_ds[i_in_shard] = eid
            idx_ds[i_in_shard] = idx
            self.append_records(h5, student_record, "student", i_in_shard, chunk0=chunk0, compression=compression)
            self.append_records(h5, teacher_record, "teacher", i_in_shard, chunk0=chunk0, compression=compression)

            i_in_shard += 1
            total_written += 1
            if (total_written % 256) == 0:  # don’t stat every sample
                h5.flush()
                if os.path.getsize(current_path) >= self.shard_size_bytes:
                    h5, current_path, shard_id, current_name, eid_ds, idx_ds, i_in_shard = self.open_new_shard(
                        h5=h5, shard_id=shard_id, shard_size=i_in_shard
                    )

        if h5 is not None:
            h5.attrs["num_samples"] = i_in_shard
            h5.flush()
            h5.close()

        return self.output_path

    # TODO: Potrebbe essere improvement fare round robin [1,1,1,1,2,2,2,2,3,3,3,3]->[1,2,3,1,2,3,1,2,3]
    #       ma vediamo poi per il momento cosi va bene. Anzi fai!

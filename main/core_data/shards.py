# Create WebDataset shards. But now we have to change structure so that:
# - Each shard contains samples from across the whole dataset (Local ds so AMIGOS only AMIGOS) randomly picked
# - For performance reasons having both teacher and student inputs -> One record has to contain both teacher student value and no longer 2 different partitions.
# - On load of a shard for first time in epoch shuffle it. Take the first B samples available (exhaustion map  to track what not to take)
import io
from pathlib import Path

import numpy as np
import pandas as pd
import tensordict
import torch
import webdataset as wds
from tensordict import TensorDict
import h5py


def materialize(x):
    if hasattr(x, "to_tensor"):  # MemoryMappedTensor
        return x.to_tensor()
    if hasattr(x, "clone"):
        return x.clone()
    return x


def to_numpy(x: torch.Tensor):
    return materialize(x).detach().cpu().numpy()


# todo log progress
class WebSharder:
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

        # caches for open memmap shards
        self.student_open = {}  # shard_name -> tensordict
        self.teacher_open = {}

    @staticmethod
    def decompose_tensordict(prefix: str, td: TensorDict) -> dict:
        decomposition = {}
        for modality, sub in td.items():
            for key, value in sub.items():
                buffer = io.BytesIO()
                torch.save(materialize(value), buffer)
                name = f"{prefix}.{modality}.{key}.pth"
                decomposition[name] = buffer.getvalue()

        return decomposition

    @staticmethod
    def get_open_shard(root: Path, cache: dict, shard_name: str):
        td = cache.get(shard_name)

        if td is None:
            td = tensordict.load_memmap(root / shard_name)  # Shard_name is a folder
            cache[shard_name] = td

        return td

    def make_ds(self):
        pass

    # First steps aggregates together the shards so that student teacher is one combo
    def to_shards(self):
        with wds.writer.ShardWriter(str(self.output_path / "ds-%06d.tar"), maxsize=self.shard_size_bytes, ) as sink:
            try:
                # todo rename che sarebbe sta roba
                cur_s_name, cur_s_td = None, None
                cur_t_name, cur_t_td = None, None
                tch_shard_td, std_shard_td = None, None

                for record in self.student_df.itertuples(index=False):
                    eid = str(record.eid)
                    idx = record.index

                    # Check for same EID and index
                    teacher_location = self.teacher_map.get((eid, idx))
                    if teacher_location is None:
                        # you can skip or raise; skipping is safer in large conversions
                        continue

                    student_shard = str(record.sharded_eid)
                    student_i = int(record.sharded_index)

                    teacher_shard, teacher_i = teacher_location
                    if student_shard != cur_s_name:
                        std_shard_td = tensordict.load_memmap(self.student_path / student_shard)
                        cur_s_name = student_shard

                    if teacher_shard != cur_t_name:
                        tch_shard_td = tensordict.load_memmap(self.teacher_path / teacher_shard)
                        cur_t_name = teacher_shard

                    student_record = std_shard_td[student_i]
                    teacher_record = tch_shard_td[teacher_i]

                    key = f"{eid}_{idx:06d}"

                    sample = {
                        "__key__": key,
                        # optionally store tiny metadata fields
                        "eid.txt": eid.encode("utf-8"),
                        "index.txt": str(idx).encode("utf-8"),
                    }

                    sample |= self.decompose_tensordict("student", student_record)
                    sample |= self.decompose_tensordict("teacher", teacher_record)

                    sink.write(sample)

            except Exception as e:
                print(e)

    # Shuffle the shards so that data is more sparse
    def shuffle(self):

        with wds.writer.ShardWriter(str(self.output_path / "s_ds-%06d.tar"), maxsize=self.shard_size_bytes, ) as sink:

            for file in Path(self.output_path).iterdir():
                if file.suffix != ".tar":
                    continue

                ds = wds.WebDataset(str(file)).decode()

    def post_check(self):
        pass  # TODO controlla che conversione corretta

    # todo valuta gzip o (meglio) .tar.zst compresion per ridurre problema di velocita di disco


class Sharder:
    def __init__(self, student_spec_path: str, teacher_spec_path: str, output_path: str, shard_size_gb=4):
        self.student_path = Path(student_spec_path).parent
        self.student_df = pd.read_csv(student_spec_path)

        self.teacher_path = Path(teacher_spec_path).parent
        self.teacher_df = pd.read_csv(teacher_spec_path)

        self.teacher_map = {
            (str(r.eid), int(r.index)): (str(r.sharded_eid), int(r.sharded_index))
            for r in self.teacher_df.itertuples(index=False)
        }

        self.output_path = Path(output_path)
        self.output_path.mkdir(parents=True, exist_ok=True)

        self.shard_size_bytes = int(shard_size_gb * (1024 ** 3))  # unused for HDF5

        # cache current open shard tensordicts
        self._cur_s_name = None
        self._cur_t_name = None
        self._std_shard_td = None
        self._tch_shard_td = None

    def _load_student_shard(self, shard_name: str):
        if shard_name != self._cur_s_name:
            self._std_shard_td = tensordict.load_memmap(self.student_path / shard_name)
            self._cur_s_name = shard_name
        return self._std_shard_td

    def _load_teacher_shard(self, shard_name: str):
        if shard_name != self._cur_t_name:
            self._tch_shard_td = tensordict.load_memmap(self.teacher_path / shard_name)
            self._cur_t_name = shard_name
        return self._tch_shard_td

    def _count_pairs(self) -> int:
        # single pass to count how many student rows have a matching teacher
        n = 0
        for r in self.student_df.itertuples(index=False):
            eid = str(r.eid)
            idx = int(r.index)
            if (eid, idx) in self.teacher_map:
                n += 1
        return n

    def _ensure_tensor_ds(self, h5: h5py.File, path: str, N: int, sample_arr: np.ndarray,
                          chunk0: int = 64, compression=None):
        """
        Create dataset once with final shape (N, ...) and chunking along axis 0.
        """
        if path in h5:
            return h5[path]

        arr = np.asarray(sample_arr)
        if arr.shape == ():
            # store scalars as (N,) not (N,1)
            shape = (N,)
            chunks = (min(chunk0, N),)
        else:
            shape = (N,) + arr.shape
            chunks = (min(chunk0, N),) + arr.shape

        return h5.create_dataset(
            path,
            shape=shape,
            dtype=arr.dtype,
            chunks=chunks,
            compression=compression,  # None (fastest) or "lzf"
        )

    def _write_record(self, h5: h5py.File, base: str, td_record, i: int, N: int,
                      chunk0: int = 64, compression=None):
        for modality, sub in td_record.items():
            for key, value in sub.items():
                arr = to_numpy(value)
                ds_path = f"{base}/{modality}/{key}"
                ds = self._ensure_tensor_ds(h5, ds_path, N, arr, chunk0=chunk0, compression=compression)
                ds[i, ...] = arr  # works for 1D and ND

    def to_hdf5(self, out_file="dataset.h5", compression=None, chunk0: int = 64):
        out_path = self.output_path / out_file

        N = self._count_pairs()

        with h5py.File(out_path, "w") as h5:
            # metadata (fixed-size now)
            str_dt = h5py.string_dtype(encoding="utf-8")
            eid_ds = h5.create_dataset("meta/eid", shape=(N,), dtype=str_dt)
            idx_ds = h5.create_dataset("meta/index", shape=(N,), dtype=np.int64)

            i = 0
            for r in self.student_df.itertuples(index=False):
                eid = str(r.eid)
                idx = int(r.index)

                teacher_loc = self.teacher_map.get((eid, idx))
                if teacher_loc is None:
                    continue

                student_shard = str(r.sharded_eid)
                student_i = int(r.sharded_index)
                teacher_shard, teacher_i = teacher_loc

                std_td = self._load_student_shard(student_shard)
                std_td["meta"].pop("experiment", None)
                tch_td = self._load_teacher_shard(teacher_shard)
                tch_td["meta"].pop("experiment", None)

                student_record = std_td[student_i]
                teacher_record = tch_td[teacher_i]

                eid_ds[i] = eid
                idx_ds[i] = idx

                self._write_record(h5, "student", student_record, i, N, chunk0=chunk0, compression=compression)
                self._write_record(h5, "teacher", teacher_record, i, N, chunk0=chunk0, compression=compression)

                i += 1

            h5.attrs["num_samples"] = i
            h5.flush()

        return out_path

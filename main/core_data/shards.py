# Create WebDataset shards. But now we have to change structure so that:
# - Each shard contains samples from across the whole dataset (Local ds so AMIGOS only AMIGOS) randomly picked
# - For performance reasons having both teacher and student inputs -> One record has to contain both teacher student value and no longer 2 different partitions.
# - On load of a shard for first time in epoch shuffle it. Take the first B samples available (exhaustion map  to track what not to take)
import io
from pathlib import Path

import pandas as pd
import tensordict
import torch
import webdataset as wds
from tensordict import TensorDict


class WebSharder:
    def __init__(self, student_spec_path: str, teacher_spec_path: str, output_path: str, shard_size_gb=4):
        self.student_path = Path(student_spec_path).parent
        self.student_df = pd.read_csv(student_spec_path)

        self.teacher_path = Path(teacher_spec_path).parent
        self.teacher_df = pd.read_csv(teacher_spec_path)

        tmap = {}
        for row in self.teacher_df.itertuples(index=False):
            # expects columns: eid, index, sharded_eid, sharded_idx
            tmap[(str(row.eid), row.index)] = (str(row.sharded_eid), int(row.sharded_idx))
        self.teacher_map = tmap

        self.output_path = output_path
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

    # First steps aggregates together the shards so that student teacher is one combo
    def make_web_shards(self):
        sink = wds.writer.ShardWriter(str(self.output_path / "ds-%06d.tar"), maxsize=self.shard_size_bytes, )
        try:
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
                student_i = int(record.sharded_idx)

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
        finally:
            sink.close()

    # Shuffle the shards so that data is more sparse
    def shuffle(self):
        pass

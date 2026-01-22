import dataclasses

import hydra

from main.core_data.shards import ReSharder


@dataclasses.dataclass
class TargetDataset:
    # Reference files
    student_spec_path: str
    teacher_spec_path: str
    # Where to store the new data
    output_path: str


@dataclasses.dataclass
class Config:
    target: TargetDataset
    shard_size_gb: int


@hydra.main(version_base=None, config_name="base", config_path="config")
def main(cfg: Config):
    sharder = ReSharder(
        student_spec_path=cfg.target.student_spec_path,
        teacher_spec_path=cfg.target.teacher_spec_path,
        output_path=cfg.target.output_path,
        shard_size_gb=cfg.shard_size_gb,
        compression="lzf"
    )

    sharder.build()


if __name__ == "__main__":
    main()
# 22151
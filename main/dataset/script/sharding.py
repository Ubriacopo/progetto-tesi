import dataclasses

import hydra

from main.core_data.shards import FusedDataSharder
from main.utils.logging import make_logger


@dataclasses.dataclass
class ShardingConfig:
    spec_path: str
    output_dir: str
    val_participants: int
    test_participants: int
    compression: str
    shard_size_gb: int


@dataclasses.dataclass
class Config:
    sharding: ShardingConfig


@hydra.main(version_base=None, config_name="sharding-local", config_path="../../../conf")
def main(cfg: Config):
    logger = make_logger("sharder")
    # todo provare
    sharder = FusedDataSharder(
        spec_path=cfg.sharding.spec_path,
        output_path=cfg.sharding.output_dir,
        val_participants=cfg.sharding.val_participants,
        test_participants=cfg.sharding.test_participants,
        compression=cfg.sharding.compression,
        shard_size_gb=cfg.sharding.shard_size_gb,
    )

    sharder.run()
    logger.info("Sharding is done!")

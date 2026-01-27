import dataclasses

import hydra

from main.core_data.shards import KdDataSharder, FusedDataSharder
from main.utils.logging import make_logger


@dataclasses.dataclass
class Config:
    pass


@hydra.main(version_base=None, config_name="sharding-local", config_path="../../../conf")
def main(cfg: Config):
    logger = make_logger("sharder")

    sharder = FusedDataSharder(

    )

    sharder.run()

    logger.info("Sharding is done!")

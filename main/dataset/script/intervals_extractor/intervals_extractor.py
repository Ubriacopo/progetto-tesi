import dataclasses
from typing import Any

import hydra
from hydra.core.config_store import ConfigStore
from hydra.utils import get_object
from omegaconf import OmegaConf

from main.core_data.extract import SegmentBasedExtractionProcessor
from main.core_data.media.text.extract import ExtractTextFromAudio
from main.core_data.media.text.transforms import WhisperExtractor
from main.dataset.utils import DatasetUidStore


@dataclasses.dataclass
class DatasetConfig:
    dataset_folder_name: str
    data_path: str
    output_path: str
    points_loader_classpath: str


@dataclasses.dataclass
class IntervalsExtractorConfig:
    base_path: str
    data_path: str
    segmenter_args: dict[str, Any]
    segmenter_type: str


@dataclasses.dataclass
class Config:
    dataset: DatasetConfig
    segmenter: IntervalsExtractorConfig
    base_path: str
    uid_store_path: str


cs = ConfigStore.instance()


@hydra.main(config_path="conf", config_name="config")
def main(cfg: Config):
    print(OmegaConf.to_yaml(cfg))
    uid_store = DatasetUidStore(cfg.uid_store_path)
    SegmentBasedExtractionProcessor(
        ExtractTextFromAudio(WhisperExtractor(model_id="openai/whisper-medium", device="cuda:0")),
        base_path=cfg.dataset.output_path,
        segmenter=get_object(cfg.segmenter.segmenter_type)(**cfg.segmenter.segmenter_args),
        loader=get_object(cfg.dataset.points_loader_classpath)(cfg.dataset.data_path, uid_store),
    ).extract_segments()


if __name__ == "__main__":
    main()

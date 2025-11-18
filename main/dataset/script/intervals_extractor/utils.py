import dataclasses
from typing import Any


@dataclasses.dataclass
class DatasetConfig:
    dataset_name: str
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

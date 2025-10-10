import dataclasses
from typing import Any

import hydra
from hydra.core.config_store import ConfigStore
from hydra.utils import get_object

from main.core_data.extract import SegmentBasedExtractionProcessor
from main.core_data.media.text.extract import ExtractTextFromAudio
from main.core_data.media.text.transforms import WhisperExtractor


@dataclasses.dataclass
class IntervalsExtractorConfig:
    base_path: str
    data_path: str
    output_path: str

    segmenter_args: dict[str, Any]
    segmenter_type: str

    points_loader_classname: str
    points_loader_classpath: str

    extract_text: bool = False


cs = ConfigStore.instance()
cs.store(name="intervals_extractor", node=IntervalsExtractorConfig)


@hydra.main(config_path="config", config_name="intervals_extractor")
def main(cfg: IntervalsExtractorConfig):
    cfg.output_path = cfg.base_path + cfg.output_path
    segmenter = get_object(cfg.segmenter_type)(**cfg.segmenter_args)

    loader = get_object(
        cfg.points_loader_classpath + "." + cfg.points_loader_classname[0].upper() + cfg.points_loader_classname[1:]
    )

    pipe = ExtractTextFromAudio(WhisperExtractor(model_id="openai/whisper-medium", device="cuda:0"))
    SegmentBasedExtractionProcessor(
        pipe, base_path=cfg.output_path, segmenter=segmenter, loader=loader(cfg.base_path + cfg.data_path),
    ).extract_segments()


if __name__ == "__main__":
    main()

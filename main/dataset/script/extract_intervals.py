import hydra
from hydra.core.config_store import ConfigStore
from hydra.utils import get_object
from omegaconf import OmegaConf

from main.core_data.extract import SegmentBasedExtractionProcessor
from main.core_data.media.text.extract import ExtractTextFromAudio
from main.core_data.media.text.transforms import WhisperExtractor

from main.dataset.utils import DatasetUidStore, PreprocessingConfig, DatasetConfig, IntervalsExtractorConfig
from main.utils.logging import make_logger

cs = ConfigStore.instance()
OmegaConf.register_new_resolver("capitalize", lambda s: s.capitalize())
OmegaConf.register_new_resolver("uppercase", lambda s: s.upper())


class Config:
    dataset: DatasetConfig
    preprocessing: PreprocessingConfig
    segmenter: IntervalsExtractorConfig

    # If the pipeline has to use the audio-2-text tool.
    transcript_audio: bool


@hydra.main(config_path="../../../conf", config_name="intervals-extractor")
def main(cfg: Config):
    logger = make_logger("intervals_extractor")
    logger.info(OmegaConf.to_yaml(cfg))
    uid_store = DatasetUidStore(cfg.dataset.uid_store_path)

    other = []
    if cfg.transcript_audio:
        other.append(ExtractTextFromAudio(WhisperExtractor(model_id="openai/whisper-medium", device="cuda:0"))),

    SegmentBasedExtractionProcessor(
        *other,
        base_path=cfg.preprocessing.extraction_data_folder,
        segmenter=get_object(cfg.segmenter.segmenter_type)(**cfg.segmenter.segmenter_args),
        loader=get_object(cfg.dataset.loader_classpath)(cfg.dataset.data_path, uid_store),
    ).extract_segments()

    logger.info(f"Done processing for {cfg.dataset.name}")


if __name__ == "__main__":
    main()

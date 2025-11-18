import hydra
from hydra.core.config_store import ConfigStore
from hydra.utils import get_object
from omegaconf import OmegaConf

from main.core_data.extract import SegmentBasedExtractionProcessor
from main.dataset.script.intervals_extractor.utils import Config
from main.dataset.utils import DatasetUidStore

cs = ConfigStore.instance()
OmegaConf.register_new_resolver("capitalize", lambda s: s.capitalize())
OmegaConf.register_new_resolver("uppercase", lambda s: s.upper())


@hydra.main(config_path="conf", config_name="config.local")
def main(cfg: Config):
    print(OmegaConf.to_yaml(cfg))
    uid_store = DatasetUidStore(cfg.uid_store_path)
    SegmentBasedExtractionProcessor(
        base_path=cfg.dataset.output_path,
        segmenter=get_object(cfg.segmenter.segmenter_type)(**cfg.segmenter.segmenter_args),
        loader=get_object(cfg.dataset.points_loader_classpath)(cfg.dataset.data_path, uid_store),
    ).extract_segments()


if __name__ == "__main__":
    main()

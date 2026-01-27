import dataclasses


@dataclasses.dataclass
class PreprocessingTargetConfig:
    out_folder_name: str
    preprocessing_pipeline: str

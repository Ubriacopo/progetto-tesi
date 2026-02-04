import dataclasses
from typing import Tuple, Optional

from main.core_data.dataset import CachableDatasetDescriptor
from main.model.neegavi.model import EegInterAviModelConfiguration


@dataclasses.dataclass
class TrainerConfig:
    lr: float
    batch_size: int
    epochs: int
    kd_loss_weight: float
    fusion_loss_weight: float
    weakly_supervised_weight: float
    ecg_correction_weight: float
    kd_temperature: float
    batches_per_epoch: int


@dataclasses.dataclass
class ModalityPresenceInformation:
    code: str  # Identification code for the modality
    cannot_miss: bool  # If it cannot miss the moment we miss it we raise an error.
    is_teacher_key: bool  # If it participates in kd process
    # Shapes of the inputs
    shape: Tuple[int, ...]
    mask_shape: Optional[Tuple[int, ...]]

    # In case it also appears in KD process
    teacher_shape: Optional[Tuple[int, ...]]
    teacher_mask_shape: Optional[Tuple[int, ...]]

    # This one is passed to the factory?
    additional_config: dict


@dataclasses.dataclass
class ModelFactoryConfig:
    factory_path: str
    args: dict


@dataclasses.dataclass
class ModelConfig:
    factory: ModelFactoryConfig

    pivot: ModalityPresenceInformation
    supports: list[ModalityPresenceInformation]

    custom_config: EegInterAviModelConfiguration


@dataclasses.dataclass
class TeacherConfig:
    hidden_channels: int
    out_channels: int

    pivot: str  # Main key of the teacher


@dataclasses.dataclass
class KdConfig:
    trainer: TrainerConfig

    model: ModelConfig
    teacher: TeacherConfig

    dataset_descriptors: list[CachableDatasetDescriptor]
    teacher_weights_path: str

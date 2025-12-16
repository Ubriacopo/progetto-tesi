import dataclasses
from typing import Tuple, Optional

import hydra
import lightning as L
import tensordict
import torch
import torchinfo
from hydra.utils import get_class
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, ConcatDataset

from main.core_data.dataset import FlexibleEmbeddingsSpecMediaDataset, RequiredKey
from main.core_data.media.assessment.assessment import Assessment
from main.core_data.media.audio import Audio
from main.core_data.media.ecg import ECG
from main.core_data.media.eeg import EEG
from main.core_data.media.text import Text
from main.core_data.media.video import Video
from main.model.VATE.constrastive_model import MaskedContrastiveModel
from main.model.kd_dataset_wrapper import KdDatasetWrapper
from main.model.neegavi.model import EegInterAviModelConfiguration
from main.model.neegavi.new_factory import AbstractEegInterAviFactory
from main.model.neegavi.train import EegAviKdVateMaskedSemiSupervisedModule


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
    classpath: str
    constructor_args: dict


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

    student_dataset_path: list[str]
    teacher_dataset_path: list[str]
    teacher_weights_path: str
# 83 108 175 119 for 42

SEED = 96


@hydra.main(config_path="config", config_name="new_train_kd")
def main(cfg: KdConfig):
    # cfg = OmegaConf.to_container(cfg, resolve=True)
    torch.manual_seed(SEED)  # Reproducibility
    factory_constructor = get_class(cfg.model.factory.classpath)
    factory = factory_constructor(**cfg.model.factory.constructor_args)

    if not isinstance(factory, AbstractEegInterAviFactory):
        raise ValueError("We need an AbstractEegInterAviFactory. Given factory is not of such type.")
    factory: AbstractEegInterAviFactory

    student = factory.build()
    teacher = MaskedContrastiveModel(hidden_channels=cfg.teacher.hidden_channels, out_channels=cfg.teacher.out_channels)

    teacher.load_state_dict(torch.load(cfg.teacher_weights_path))
    teacher.eval()

    fusion_metrics_codes = [cfg.model.supports[s].code for s in cfg.model.supports]
    fusion_metrics_codes.append(cfg.model.pivot.code)

    module = EegAviKdVateMaskedSemiSupervisedModule(
        student=student,
        teacher=teacher,
        kd_loss_weight=cfg.trainer.kd_loss_weight,
        fusion_loss_weight=cfg.trainer.fusion_loss_weight,
        weakly_supervised_weight=cfg.trainer.weakly_supervised_weight,
        lr=cfg.trainer.lr,
        kd_temperature=cfg.trainer.kd_temperature,
        # All modalities contribute to fusion
        fusion_metrics=fusion_metrics_codes,
    )

    student_keys: list[RequiredKey] = []
    teacher_keys: list[RequiredKey] = []

    # Pivot
    c = cfg.model.pivot
    student_keys.append(RequiredKey(c.code, c.shape, c.mask_shape, c.cannot_miss))
    if c.is_teacher_key:
        teacher_keys.append(RequiredKey(c.code, c.teacher_shape, c.teacher_mask_shape, c.cannot_miss))

    # Supports
    for key in cfg.model.supports:
        c = cfg.model.supports[key]
        student_keys.append(RequiredKey(c.code, c.shape, c.mask_shape, c.cannot_miss))
        if c.is_teacher_key:
            teacher_keys.append(RequiredKey(c.code, c.teacher_shape, c.teacher_mask_shape, c.cannot_miss))

    student_dataset = []
    for file in cfg.student_dataset_path:
        pivot_key = cfg.model.pivot.code
        ds = FlexibleEmbeddingsSpecMediaDataset(dataset_spec_file=file, required_keys=student_keys, main_key=pivot_key)
        student_dataset.append(ds)

    student_dataset = ConcatDataset(student_dataset)

    teacher_dataset = []
    for file in cfg.teacher_dataset_path:
        pivot_key = cfg.teacher.pivot
        ds = FlexibleEmbeddingsSpecMediaDataset(dataset_spec_file=file, required_keys=teacher_keys, main_key=pivot_key)
        teacher_dataset.append(ds)

    teacher_dataset = ConcatDataset(teacher_dataset)

    def collate_fn(batch):
        return tensordict.stack(batch)

    dataset_wrapper = KdDatasetWrapper(student=student_dataset, teacher=teacher_dataset)
    train_dataloader = DataLoader(
        dataset_wrapper, batch_size=cfg.trainer.batch_size, shuffle=True, collate_fn=collate_fn
    )

    for n, p in student.named_parameters():
        print(n, p.requires_grad, p.grad is None)

    torchinfo.summary(module)
    trainer = L.Trainer(accelerator="gpu", devices=1, max_epochs=cfg.trainer.epochs, log_every_n_steps=24, limit_train_batches=1)
    #trainer = L.Trainer(accelerator="gpu", devices=1, max_epochs=cfg.trainer.epochs, log_every_n_steps=24)
    trainer.fit(module, train_dataloader)


if __name__ == "__main__":
    main()

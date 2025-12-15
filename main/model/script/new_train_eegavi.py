import dataclasses

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
class ModelFactoryConfig:
    classpath: str
    constructor: dict

    pivot_code: str
    supports_codes: list[str]


@dataclasses.dataclass
class KdConfig:
    trainer: TrainerConfig
    factory: ModelFactoryConfig  # TODO vedi se riesci a fare type hints

    student_dataset_path: list[str]
    teacher_dataset_path: list[str]
    teacher_weights_path: str


SEED = 42


@hydra.main(config_path="config", config_name="new_train_kd")
def main(cfg: KdConfig):
    # cfg = OmegaConf.to_container(cfg, resolve=True)
    torch.manual_seed(SEED)  # Reproducibility
    factory_constructor = get_class(cfg.factory.classpath)
    factory = factory_constructor(**cfg.factory.constructor)

    if not isinstance(factory, AbstractEegInterAviFactory):
        raise ValueError("We need an AbstractEegInterAviFactory. Given factory is not of such type.")
    factory: AbstractEegInterAviFactory

    student = factory.build()
    teacher = MaskedContrastiveModel(hidden_channels=200, out_channels=100)

    teacher.load_state_dict(torch.load(cfg.teacher_weights_path))
    teacher.eval()

    module = EegAviKdVateMaskedSemiSupervisedModule(
        student=student,
        teacher=teacher,
        kd_loss_weight=cfg.trainer.kd_loss_weight,
        fusion_loss_weight=cfg.trainer.fusion_loss_weight,
        weakly_supervised_weight=cfg.trainer.weakly_supervised_weight,
        lr=cfg.trainer.lr,
        kd_temperature=cfg.trainer.kd_temperature,
        # All modalities contribute to fusion
        fusion_metrics=cfg.factory.supports_codes + [cfg.factory.pivot_code],
    )
    # todo da leggere da config
    student_keys: list[RequiredKey] = [
        RequiredKey(EEG.modality_code(), shape=(8, 32, 34, 256), mask_shape=(8, 32, 34), cannot_miss=True),
        # RequiredKey("meta", ), Cannot recreate this.
        RequiredKey(Assessment.modality_code(), shape=(4,), mask_shape=(4,)),
        RequiredKey(Video.modality_code(), shape=(8, 16, 768), mask_shape=(8,)),
        RequiredKey(Audio.modality_code(), shape=(8, 199, 768), mask_shape=(8,)),
        RequiredKey(ECG.modality_code(), shape=(8, 32, 256), mask_shape=(8,)),
        RequiredKey(Text.modality_code(), shape=(8, 384), mask_shape=(8,))
    ]

    student_dataset = ConcatDataset([
        FlexibleEmbeddingsSpecMediaDataset(
            dataset_spec_file=file, cache_in_ram=True, required_keys=student_keys, main_key=EEG.modality_code()
        )
        for file in cfg.student_dataset_path
    ])

    # todo da leggere da config
    teacher_keys: list[RequiredKey] = [
        RequiredKey(Video.modality_code(), shape=(400,), mask_shape=(1,), cannot_miss=True),
        RequiredKey(Audio.modality_code(), shape=(768,), mask_shape=(1,)),
        RequiredKey(Text.modality_code(), shape=(768,), mask_shape=(1,))
    ]

    teacher_dataset = ConcatDataset([
        FlexibleEmbeddingsSpecMediaDataset(
            dataset_spec_file=file, cache_in_ram=True, required_keys=teacher_keys, main_key=Video.modality_code()
        )
        for file in cfg.teacher_dataset_path
    ])

    dataset_wrapper = KdDatasetWrapper(student=student_dataset, teacher=teacher_dataset)
    train_dataloader = DataLoader(
        dataset_wrapper, batch_size=cfg.batch_size, shuffle=True, collate_fn=lambda x: tensordict.stack(x)
    )

    for n, p in student.named_parameters():
        print(n, p.requires_grad, p.grad is None)

    torchinfo.summary(module)
    # trainer = L.Trainer(accelerator="gpu", devices=1, max_epochs=cfg.epochs, log_every_n_steps=24, overfit_batches=1)
    trainer = L.Trainer(accelerator="gpu", devices=1, max_epochs=cfg.epochs, log_every_n_steps=24)
    trainer.fit(module, train_dataloader)


if __name__ == "__main__":
    main()

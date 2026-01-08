import dataclasses
from typing import Tuple, Optional

import hydra
import lightning as L
import tensordict
import torch
import torchinfo
from hydra.utils import get_class
from lightning.pytorch.profilers import SimpleProfiler
from torch.utils.data import DataLoader

from main.core_data.dataset import RequiredKey, MultiDataset, \
    DatasetFirstBatchSampler, SequentialPerDatasetBatchSampler, MultiDatasetQueueBatchSampler, \
    FlexibleEmbeddingsSpecMediaDatasetSlow
from main.model.VATE.constrastive_model import MaskedContrastiveModel
from main.model.kd_dataset_wrapper import KdDatasetWrapper
from main.model.neegavi.factory import AbstractEegInterAviFactory
from main.model.neegavi.model import EegInterAviModelConfiguration
from main.model.neegavi.train import EegAviKdVateMaskedSemiSupervisedModule
import torch.multiprocessing as mp

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


@hydra.main(config_path="config", config_name="default")
def main(cfg: KdConfig):
    # cfg = OmegaConf.to_container(cfg, resolve=True)
    torch.manual_seed(SEED)  # Reproducibility
    factory_constructor = get_class(cfg.model.factory.classpath)
    factory = factory_constructor(**cfg.model.factory.constructor_args)
    if not isinstance(factory, AbstractEegInterAviFactory):
        raise ValueError("We need an AbstractEegInterAviFactory. Given factory is not of such type.")

    factory: AbstractEegInterAviFactory

    student = factory.build()

    # Teacher construction
    teacher = MaskedContrastiveModel(hidden_channels=cfg.teacher.hidden_channels, out_channels=cfg.teacher.out_channels)
    teacher.load_state_dict(torch.load(cfg.teacher_weights_path))
    teacher.eval()  # Set to evaluation mode as we won't be learning on teacher.

    fusion_metrics_codes = [cfg.model.supports[s].code for s in cfg.model.supports]
    fusion_metrics_codes.append(cfg.model.pivot.code)

    c = cfg.model.pivot
    student_keys: list[RequiredKey] = [
        RequiredKey(c.code, c.shape, c.mask_shape, c.cannot_miss)
    ]

    teacher_keys: list[RequiredKey] = []
    if c.is_teacher_key:
        teacher_keys = [RequiredKey(c.code, c.teacher_shape, c.teacher_mask_shape, c.cannot_miss)]

    # Each support has to be registered as key in student and also at times in teacher.
    for key in cfg.model.supports:
        c = cfg.model.supports[key]
        student_keys.append(RequiredKey(c.code, c.shape, c.mask_shape, c.cannot_miss))
        if c.is_teacher_key:
            teacher_keys.append(RequiredKey(c.code, c.teacher_shape, c.teacher_mask_shape, c.cannot_miss))

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
        kd_keys=list(map(lambda o: o.key, teacher_keys))
    )

    dataset_pairs = []
    for student_file, teacher_file in zip(cfg.student_dataset_path, cfg.teacher_dataset_path):
        pivot_key = cfg.model.pivot.code
        dataset_pairs.append(KdDatasetWrapper(
            student=FlexibleEmbeddingsSpecMediaDatasetSlow(student_file, student_keys, main_key=pivot_key,selected_device='cpu'),
            teacher=FlexibleEmbeddingsSpecMediaDatasetSlow(
                teacher_file, teacher_keys, main_key=cfg.teacher.pivot, squeeze_mask=True, selected_device='cpu'
            )
        ))

    g = torch.Generator().manual_seed(SEED)
    # Partition the dataset in 3 splits (Percentage should per parameter of Trainer)
    train_dataset, valid_dataset, test_dataset = MultiDataset(dataset_pairs).split(0.75, 0.15, seed=SEED)
    # todo: see how many samples all ds have

    # todo parameterize
    batch_sampler = DatasetFirstBatchSampler(
        multi=train_dataset,
        batch_size=cfg.trainer.batch_size,
        batches_per_epoch=cfg.trainer.batches_per_epoch,  # you choose
        alpha=0.0,
        generator=g,
    )

    def collate_fn(batch):
        return tensordict.stack(batch)

    # In case overfit experiment
    batch_size = cfg.trainer.batch_size
    if False:
        batch_sampler = [next(iter(batch_sampler))]  # grab one batch

    train_dataloader = DataLoader(
        train_dataset,
        # TODO parametrize
        batch_sampler=MultiDatasetQueueBatchSampler(
            multi=train_dataset,
            batch_size=cfg.trainer.batch_size,
            batches_per_epoch=cfg.trainer.batches_per_epoch,  # you choose
            alpha=0.0,
            generator=g,
        ),
        collate_fn=collate_fn,
        num_workers=12,
        prefetch_factor=8,
        persistent_workers=True,
    )

    valid_dataloader = DataLoader(
        valid_dataset,
        batch_sampler=SequentialPerDatasetBatchSampler(multi=valid_dataset, batch_size=cfg.trainer.batch_size, ),
        shuffle=None, collate_fn=collate_fn,
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_sampler=SequentialPerDatasetBatchSampler(multi=test_dataset, batch_size=cfg.trainer.batch_size, ),
        collate_fn=collate_fn,
        num_workers=0,
    )

    for n, p in student.named_parameters():
        print(n, p.requires_grad, p.grad is None)

    torchinfo.summary(module)
    trainer = L.Trainer(
        profiler=SimpleProfiler(),
        accelerator="gpu",
        devices=1,
        max_epochs=cfg.trainer.epochs,
        callbacks=[
            # TQDMProgressBar(leave=True, refresh_rate=40)
        ],
        num_sanity_val_steps=0,
        # precision="16-mixed", P6000 has no tensor cores
        log_every_n_steps=50,
        enable_progress_bar=False
        # limit_train_batches=1
    )
    # trainer = L.Trainer(accelerator="gpu", devices=1, max_epochs=cfg.trainer.epochs, log_every_n_steps=24)
    trainer.fit(module, train_dataloaders=train_dataloader)
    # trainer.fit(module, train_dataloaders=train_dataloader, val_dataloaders=valid_dataloader, )

    print(trainer.profiler.summary())


if __name__ == "__main__":
    main()

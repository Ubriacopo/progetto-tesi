import dataclasses
from typing import Tuple, Optional

import hydra
import lightning as L
import tensordict
import torch
import torchinfo
from hydra.utils import get_class
from lightning.pytorch.callbacks import TQDMProgressBar
from torch.utils.data import DataLoader, Subset

from main.core_data.dataset import FlexibleEmbeddingsSpecMediaDataset, RequiredKey, MultiDataset, \
    DatasetFirstBatchSampler
from main.model.VATE.constrastive_model import MaskedContrastiveModel
from main.model.kd_dataset_wrapper import KdDatasetWrapper
from main.model.neegavi.factory import AbstractEegInterAviFactory
from main.model.neegavi.model import EegInterAviModelConfiguration
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


def split_multidataset(md: MultiDataset, train=0.75, val=0.10, seed=0):
    g = torch.Generator().manual_seed(seed)

    train_parts, val_parts, test_parts = [], [], []
    for ds in md.datasets:
        n = len(ds)
        perm = torch.randperm(n, generator=g).tolist()

        n_train = int(train * n)
        n_val = int(val * n)

        train_idx = perm[:n_train]
        val_idx = perm[n_train:n_train + n_val]
        test_idx = perm[n_train + n_val:]

        train_parts.append(Subset(ds, train_idx))
        val_parts.append(Subset(ds, val_idx))
        test_parts.append(Subset(ds, test_idx))

    return MultiDataset(train_parts), MultiDataset(val_parts), MultiDataset(test_parts)


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
            student=FlexibleEmbeddingsSpecMediaDataset(student_file, student_keys, main_key=pivot_key),
            teacher=FlexibleEmbeddingsSpecMediaDataset(
                teacher_file, teacher_keys, main_key=cfg.teacher.pivot, squeeze_mask=True
            )
        ))

    g = torch.Generator().manual_seed(SEED)
    full_dataset = MultiDataset(dataset_pairs)
    # Partition the dataset in 3 splits (Percentage should per parameter of Trainer)
    train_dataset, valid_dataset, test_dataset = split_multidataset(full_dataset, seed=SEED)

    # todo parameterize
    batch_sampler = DatasetFirstBatchSampler(
        multi=train_dataset,
        batch_size=cfg.trainer.batch_size,
        batches_per_epoch=100,  # you choose
        alpha=0.0,
        generator=g,
    )
    indices = next(iter(batch_sampler))  # grab one batch

    def collate_fn(batch):
        return tensordict.stack(batch)

    # In case overfit experiment
    batch_size = cfg.trainer.batch_size
    if True:
        batch_sampler = [next(iter(batch_sampler))]  # grab one batch

    train_dataloader = DataLoader(train_dataset, batch_sampler=batch_sampler, collate_fn=collate_fn)
    # train_dataloader = DataLoader(train_dataset, batch_sampler=batch_sampler, collate_fn=collate_fn)
    valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    for n, p in student.named_parameters():
        print(n, p.requires_grad, p.grad is None)

    torchinfo.summary(module)
    trainer = L.Trainer(
        accelerator="gpu",
        devices=1,
        max_epochs=cfg.trainer.epochs,
        log_every_n_steps=24,
        callbacks=[TQDMProgressBar(leave=True)],
        limit_train_batches=1
    )
    # trainer = L.Trainer(accelerator="gpu", devices=1, max_epochs=cfg.trainer.epochs, log_every_n_steps=24)
    trainer.fit(module, train_dataloaders=train_dataloader, val_dataloaders=valid_dataloader, )


if __name__ == "__main__":
    main()

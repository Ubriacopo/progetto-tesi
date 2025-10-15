import dataclasses

import hydra
import lightning as L
import torch
from torch.utils.data import DataLoader, ConcatDataset
from torchview import draw_graph

from main.core_data.dataset import FlexibleEmbeddingsSpecMediaDataset
from main.model.EEGAVI.factory import EegBaseModelFactory
from main.model.EEGAVI.interleaved_EEGAVI.interleaved_model import get_interleaved_EEG_AVI, \
    get_interleaved_weakly_supervised
from main.model.VATE.constrastive_model import MaskedContrastiveModel
from main.model.kd_dataset_wrapper import KdDatasetWrapper
from main.model.kd_train import EegAviKdVateMaskedSemiSupervisedModule


@dataclasses.dataclass
class KdConfig:
    base_path: str
    use_base_path: bool

    student_dataset_path: list[str]
    teacher_dataset_path: list[str]
    teacher_weights_path: str

    # Args for training.
    lr: float
    batch_size: int
    epochs: int

    # Args for loss computation (weighting)
    kd_loss_weight: float
    fusion_loss_weight: float
    weakly_supervised_weight: float
    ecg_correction_weight: float
    kd_temperature: float


SEED = 42


@hydra.main(config_path="config", config_name="train_kd")
def main(cfg: KdConfig):
    if cfg.use_base_path:
        for i in range(cfg.student_dataset_path.__len__()):
            cfg.student_dataset_path[i] = cfg.base_path + cfg.student_dataset_path[i]
            cfg.teacher_dataset_path[i] = cfg.base_path + cfg.teacher_dataset_path[i]
        cfg.teacher_weights_path = cfg.base_path + cfg.teacher_weights_path

    torch.manual_seed(SEED)  # Reproducibility
    student = EegBaseModelFactory.weak_supervised_interleaved(
        output_size=384, base_model_target_size=384, supports_latent_size=384
    )
    teacher = MaskedContrastiveModel(hidden_channels=200, out_channels=100)

    teacher.load_state_dict(torch.load(cfg.teacher_weights_path))
    teacher.eval()
    module = EegAviKdVateMaskedSemiSupervisedModule(
        student=student,
        teacher=teacher,
        kd_loss_weight=cfg.kd_loss_weight,
        fusion_loss_weight=cfg.fusion_loss_weight,
        weakly_supervised_weight=cfg.weakly_supervised_weight,
        ecg_correction_weight=cfg.ecg_correction_weight,
        lr=cfg.lr,
        kd_temperature=cfg.kd_temperature,
    )

    student_dataset = ConcatDataset([
        FlexibleEmbeddingsSpecMediaDataset(dataset_spec_file=file, cache_in_ram=True)
        for file in cfg.student_dataset_path
    ])

    teacher_dataset = ConcatDataset([
        FlexibleEmbeddingsSpecMediaDataset(dataset_spec_file=file, cache_in_ram=True)
        for file in cfg.teacher_dataset_path
    ])

    dataset_wrapper = KdDatasetWrapper(student=student_dataset, teacher=teacher_dataset)
    train_dataloader = DataLoader(dataset_wrapper, batch_size=cfg.batch_size, shuffle=True)
    # Plot trained model structure and store to file
    # model_graph = draw_graph(
    #    student.eeg_avi,
    #    input_data={"x": next(iter(train_dataloader))["student"], "use_kd": True},
    #    filename="student_model",
    #    directory="structure",
    #    save_graph=True,
    #    depth=1,
    #    expand_nested=True,
    #    hide_module_functions=True
    # )

    trainer = L.Trainer(accelerator="gpu", devices=1, max_epochs=cfg.epochs, log_every_n_steps=24, overfit_batches=1)
    trainer.fit(module, train_dataloader)


if __name__ == "__main__":
    main()

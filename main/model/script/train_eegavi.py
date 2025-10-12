import dataclasses

import hydra
import torch
from lightning.pytorch.utilities import CombinedLoader
from torch.utils.data import DataLoader
import lightning as L

from main.core_data.dataset import FlexibleEmbeddingsSpecMediaDataset
from main.model.EEGAVI.interleaved_EEGAVI.interleaved_model import get_interleaved_EEG_AVI
from main.model.EEGAVI.kd_module import EegAviKdVateMaskedModule
from main.model.VATE.constrastive_model import MaskedContrastiveModel


@dataclasses.dataclass
class KdConfig:
    base_path: str

    batch_size: int
    student_dataset_path: str
    teacher_dataset_path: str
    teacher_weights_path: str

    use_base_path: bool = True


SEED = 42


@hydra.main(config_path="config", config_name="prepare_dataset_config_pre_extracted")
def main(cfg: KdConfig):
    if cfg.use_base_path:
        cfg.student_dataset_path = cfg.base_path + cfg.student_dataset_path
        cfg.teacher_dataset_path = cfg.base_path + cfg.teacher_dataset_path
        cfg.teacher_weights_path = cfg.base_path + cfg.teacher_weights_path

    torch.manual_seed(SEED)
    student = get_interleaved_EEG_AVI(target_size=384, supporting_latent_size=384)
    teacher = MaskedContrastiveModel(hidden_channels=200, out_channels=100)
    # "../../dependencies/VATE/best_model_contrastive.pt"
    teacher.load_state_dict(torch.load(cfg.teacher_weights_path))
    teacher.eval()
    module = EegAviKdVateMaskedModule(student, teacher)

    # "../../../data/amigos/p-interleaved-d/spec.csv"
    student_dataset = FlexibleEmbeddingsSpecMediaDataset(dataset_spec_file=cfg.student_dataset_path, cache_in_ram=True)
    idx = torch.randperm(len(student_dataset)).tolist()  # TODO non va bene no shuffle!
    teacher_dataste = FlexibleEmbeddingsSpecMediaDataset(dataset_spec_file=cfg.teacher_dataset_path, cache_in_ram=True)
    # todo wrapper dataset teacher-student cosi evito problemi di datalaoder
    train_dataloader = CombinedLoader({
        "student": DataLoader(student_dataset, batch_size=cfg.batch_size, sampler=idx[:cfg.batch_size], shuffle=False),
        "teacher": DataLoader(teacher_dataste, batch_size=cfg.batch_size, sampler=idx[cfg.batch_size:], shuffle=False),
    })

    trainer = L.Trainer(accelerator="gpu", devices=1, max_epochs=10)
    trainer.fit(module, train_dataloader)

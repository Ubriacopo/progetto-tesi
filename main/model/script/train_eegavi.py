import dataclasses

import hydra
import torch
from torch.utils.data import DataLoader, ConcatDataset
import lightning as L

from main.core_data.dataset import FlexibleEmbeddingsSpecMediaDataset
from main.model.EEGAVI.interleaved_EEGAVI.interleaved_model import get_interleaved_EEG_AVI
from main.model.EEGAVI.kd_module import EegAviKdVateMaskedModule
from main.model.VATE.constrastive_model import MaskedContrastiveModel
from main.model.kd_dataset_wrapper import KdDatasetWrapper


@dataclasses.dataclass
class KdConfig:
    base_path: str

    batch_size: int
    student_dataset_path: list[str]
    teacher_dataset_path: list[str]
    teacher_weights_path: str

    use_base_path: bool = True


SEED = 42

# todo far partire scritp
@hydra.main(config_path="config", config_name="prepare_dataset_config_pre_extracted")
def main(cfg: KdConfig):
    if cfg.use_base_path:
        for i in range(cfg.student_dataset_path.__len__()):
            cfg.student_dataset_path[i] = cfg.base_path + cfg.student_dataset_path[i]
            cfg.teacher_dataset_path[i] = cfg.base_path + cfg.teacher_dataset_path[i]
        cfg.teacher_weights_path = cfg.base_path + cfg.teacher_weights_path

    torch.manual_seed(SEED)
    student = get_interleaved_EEG_AVI(target_size=384, supporting_latent_size=384)
    teacher = MaskedContrastiveModel(hidden_channels=200, out_channels=100)
    # "../../dependencies/VATE/best_model_contrastive.pt"
    teacher.load_state_dict(torch.load(cfg.teacher_weights_path))
    teacher.eval()
    module = EegAviKdVateMaskedModule(student, teacher)

    # "../../../data/amigos/p-interleaved-d/spec.csv"

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

    trainer = L.Trainer(accelerator="gpu", devices=1, max_epochs=10)
    trainer.fit(module, train_dataloader)

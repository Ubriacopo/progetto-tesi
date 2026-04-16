import torch

from main.model.downstream.core.probe_model import SimpleCbraFineTune
from main.model.downstream.core.trainer.base import AbstractClassificationTrainer


class FacedTrainer(AbstractClassificationTrainer):
    def extract_target(self, batch):
        return batch["assessment", "score"][:, 0]


class CBraModFacedClassificationTrainer(AbstractClassificationTrainer):
    def __init__(self, model: SimpleCbraFineTune, seed: int, lr=3e-4, classes: int = 2,
                 backbone_lr=3e-4, cbramod_lr=1e-5, weight_decay: float = 0.01):
        super().__init__(model, seed, lr, classes, backbone_lr)

    def extract_target(self, batch):
        return batch["assessment", "score"][:, 0]

    def configure_optimizers(self):
        m: SimpleCbraFineTune = self.model
        project_params = m.project.parameters()
        cbra_params = [p for p in m.encoder.parameters() if p.requires_grad]

        optimizer = torch.optim.AdamW([
            {"params": project_params, "lr": self.hparams.lr},
            {"params": cbra_params, "lr": self.hparams.backbone_lr},
        ], weight_decay=self.hparams.weight_decay, )

        return optimizer

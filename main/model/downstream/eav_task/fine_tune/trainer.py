import torch
from torch import nn

from main.model.downstream.core.trainer.base import AbstractClassificationTrainer


class EavClassificationTrainer(AbstractClassificationTrainer):
    def __init__(self, model: nn.Module, seed: int, lr=3e-4, classes: int = 2, backbone_lr=3e-5):
        super().__init__(model, seed, lr, classes, backbone_lr)

    def extract_target(self, batch):
        return batch["assessment", "score"][:, 0]

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW([
            {"params": self.model.project.parameters(), "lr": self.hparams.lr},
            {"params": filter(lambda p: p.requires_grad, self.model.backbone.parameters()), "lr": self.hparams.backbone_lr},
        ])

        return optimizer

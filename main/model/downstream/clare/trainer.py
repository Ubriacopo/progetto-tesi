import torch
from torch import nn

from main.model.downstream.core.trainer.base import AbstractEegAviClassificationTrainer, \
    AbstractCBraClassificationTrainer


class ClaireClassificationTrainer(AbstractEegAviClassificationTrainer):
    def __init__(self, model: nn.Module, seed: int, lr=3e-4, classes: int = 2, backbone_lr=3e-5):
        super().__init__(model, seed, classes, lr, backbone_lr)
        self.register_buffer("class_weights", torch.tensor([1.59, 0.73], dtype=torch.float32))
        self.criterion = torch.nn.CrossEntropyLoss(weight=self.class_weights)

    def extract_target(self, batch):
        return (batch["assessment", "scores"].float() >= 5.).long()


class CBraModClareClassificationTrainer(AbstractCBraClassificationTrainer):
    def __init__(self, model: nn.Module, seed: int, lr=3e-4, classes: int = 2, backbone_lr=3e-5):
        super().__init__(model, seed, lr, classes, backbone_lr)
        self.register_buffer("class_weights", torch.tensor([1.59, 0.73], dtype=torch.float32))
        self.criterion = torch.nn.CrossEntropyLoss(weight=self.class_weights)

    def extract_target(self, batch):
        return (batch["assessment", "scores"].float() >= 5.).long()

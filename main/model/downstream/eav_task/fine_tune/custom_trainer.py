import torch

from main.model.downstream.core.trainer.classification import ClassificationTrainer


class EavFineTuneTrainer(ClassificationTrainer):
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW([
            {"params": self.model.project.parameters(), "lr": self.hparams.lr},
            {"params": filter(lambda p: p.requires_grad, self.model.backbone.parameters()), "lr": self.hparams.backbone_lr},
        ])

        return optimizer

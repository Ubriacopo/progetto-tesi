from abc import ABC, abstractmethod

import lightning
import torch
from lightning.pytorch.utilities.types import OptimizerLRScheduler
from torch import nn
from torchmetrics.classification import MulticlassConfusionMatrix, MulticlassF1Score, MulticlassAccuracy

from main.core_data.media.audio import Audio
from main.core_data.media.ecg import ECG
from main.core_data.media.eeg import EEG
from main.core_data.media.video import Video
from main.model.downstream.core.utils import dequantize


class AbstractClassificationTrainer(lightning.LightningModule, ABC):
    default_dequantize_keys = [EEG.modality_code(), Video.modality_code(), Audio.modality_code(), ECG.modality_code()]

    def __init__(self, model: nn.Module, seed: int, lr=3e-4, classes: int = 2, backbone_lr=3e-5):
        super().__init__()
        self.model: nn.Module = model
        self.save_hyperparameters(ignore=["model"])

        #self.register_buffer("class_weights", torch.tensor([1.39, 3.57], dtype=torch.float32))
        #self.criterion = torch.nn.CrossEntropyLoss(weight=self.class_weights)
        self.criterion = torch.nn.CrossEntropyLoss()
        # validation metrics
        self.val_acc = MulticlassAccuracy(num_classes=classes, average="micro")
        self.val_bal_acc = MulticlassAccuracy(num_classes=classes, average="macro")
        self.val_f1 = MulticlassF1Score(num_classes=classes, average="macro")
        self.val_cm = MulticlassConfusionMatrix(num_classes=classes)
        # test metrics
        self.test_acc = MulticlassAccuracy(num_classes=classes, average="micro")
        self.test_bal_acc = MulticlassAccuracy(num_classes=classes, average="macro")
        self.test_f1 = MulticlassF1Score(num_classes=classes, average="macro")
        self.test_cm = MulticlassConfusionMatrix(num_classes=classes)

    def configure_optimizers(self) -> OptimizerLRScheduler:
        optimizer = torch.optim.AdamW([
            {"params": self.model.project.parameters(), "lr": self.hparams.lr},
            #  {"params": filter(lambda p: p.requires_grad, self.model.backbone.parameters()), "lr": self.hparams.backbone_lr},
        ])

        return optimizer

    @abstractmethod
    def extract_target(self, batch):
        pass

    def training_step(self, batch):
        # Labels
        y = self.extract_target(batch)
        x = dequantize(batch, self.default_dequantize_keys)
        pred = self.model(x)

        loss = self.criterion(pred, y)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch):
        y = self.extract_target(batch)
        x = dequantize(batch, self.default_dequantize_keys)
        pred = self.model(x)

        loss = self.criterion(pred, y)
        self.log("val_loss", loss, prog_bar=True)

        self.val_acc.update(pred, y)
        self.val_bal_acc.update(pred, y)
        self.val_f1.update(pred, y)
        self.val_cm.update(pred, y)

        self.log("val_acc", self.val_acc, prog_bar=True, on_step=False, on_epoch=True)
        self.log("val_bal_acc", self.val_bal_acc, prog_bar=True, on_step=False, on_epoch=True)
        self.log("val_f1", self.val_f1, prog_bar=True, on_step=False, on_epoch=True)

        return loss

    def test_step(self, batch):
        y = self.extract_target(batch)
        x = dequantize(batch, self.default_dequantize_keys)
        pred = self.model(x)

        loss = self.criterion(pred, y)
        self.log("test_loss", loss, prog_bar=True)

        self.test_acc.update(pred, y)
        self.test_bal_acc.update(pred, y)
        self.test_f1.update(pred, y)
        self.test_cm.update(pred, y)

        self.log("test_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log("test_acc", self.test_acc, prog_bar=True, on_step=False, on_epoch=True)
        self.log("test_bal_acc", self.test_bal_acc, prog_bar=True, on_step=False, on_epoch=True)
        self.log("test_f1", self.test_f1, prog_bar=True, on_step=False, on_epoch=True)

        return loss

    def on_test_epoch_end(self):
        cm = self.test_cm.compute()
        print("\nTest confusion matrix:")
        print(cm)
        self.test_cm.reset()

    def on_validation_epoch_end(self):
        cm = self.val_cm.compute()
        print("\nValidation confusion matrix:")
        print(cm)
        self.val_cm.reset()

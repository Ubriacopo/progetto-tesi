from abc import ABC, abstractmethod

import lightning
import torch
import torch.nn.functional as F
from lightning.pytorch.utilities.types import OptimizerLRScheduler
from torch import nn
from torchmetrics import R2Score, MeanSquaredError, MeanAbsoluteError

from main.core_data.media.audio import Audio
from main.core_data.media.ecg import ECG
from main.core_data.media.eeg import EEG
from main.core_data.media.video import Video
from main.model.downstream.core.utils import dequantize


class RegressionTrainer(lightning.LightningModule, ABC):
    default_dequantize_keys = [EEG.modality_code(), Video.modality_code(), Audio.modality_code(), ECG.modality_code()]

    def __init__(self, model: nn.Module, seed: int, lr=3e-4, backbone_lr=3e-5):
        super().__init__()
        self.model: nn.Module = model
        self.save_hyperparameters(ignore=["model"])
        # Train metrics
        self.train_rmse = MeanSquaredError(squared=False)
        # Validation metrics
        self.val_rmse = MeanSquaredError(squared=False)
        self.val_mae = MeanAbsoluteError()
        # Test metrics
        self.test_rmse = MeanSquaredError(squared=False)
        self.test_mae = MeanAbsoluteError()


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
        pred = self.model(x).squeeze()

        loss = F.mse_loss(pred, y)
        self.train_rmse.update(pred, y)
        self.log("train_loss", loss, prog_bar=True, batch_size=y.numel())
        self.log("train_rmse", self.train_rmse, prog_bar=True)
        return loss


    def validation_step(self, batch):
        y = self.extract_target(batch)
        x = dequantize(batch, self.default_dequantize_keys)
        pred: torch.Tensor = self.model(x).squeeze()

        loss = F.mse_loss(pred, y)

        self.val_rmse.update(pred, y)
        self.val_mae.update(pred, y)

        self.log("val_loss", loss, prog_bar=True, batch_size=y.numel())
        self.log("val_rmse", self.val_rmse, prog_bar=True)
        self.log("val_mae", self.val_mae, prog_bar=True, on_step=False, on_epoch=True)
        return loss


    def test_step(self, batch):
        y = self.extract_target(batch)
        x = dequantize(batch, self.default_dequantize_keys)
        pred = self.model(x).squeeze()

        loss = F.mse_loss(pred, y)

        self.test_rmse.update(pred, y)
        self.test_mae.update(pred, y)

        self.log("test_loss", loss, prog_bar=True, batch_size=y.numel())
        self.log("test_rmse", self.test_rmse, prog_bar=True)
        self.log("test_mae", self.test_mae, prog_bar=True, on_step=False, on_epoch=True)
        return loss

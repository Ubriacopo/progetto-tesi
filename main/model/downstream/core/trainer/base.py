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
from main.model.downstream.core.model.finetune import EegAviFineTune, CBraFineTune
from main.model.downstream.core.utils import dequantize
from main.model.neegavi.adapters import EegCbraModAdapter


class AbstractClassificationTrainer(lightning.LightningModule, ABC):
    default_dequantize_keys = [EEG.modality_code(), Video.modality_code(), Audio.modality_code(), ECG.modality_code()]

    def __init__(self, model: nn.Module, seed: int, lr=3e-4, classes: int = 2, backbone_lr=3e-5):
        super().__init__()
        self.model: nn.Module = model
        self.save_hyperparameters(ignore=["model"])
        self.criterion = torch.nn.CrossEntropyLoss()
        self.train_acc = MulticlassAccuracy(num_classes=classes, average="micro")
        # Validation metrics
        self.val_acc = MulticlassAccuracy(num_classes=classes, average="micro")
        self.val_bal_acc = MulticlassAccuracy(num_classes=classes, average="macro")
        self.val_f1 = MulticlassF1Score(num_classes=classes, average="macro")
        self.val_cm = MulticlassConfusionMatrix(num_classes=classes)
        # Test metrics
        self.test_acc = MulticlassAccuracy(num_classes=classes, average="micro")
        self.test_bal_acc = MulticlassAccuracy(num_classes=classes, average="macro")
        self.test_f1 = MulticlassF1Score(num_classes=classes, average="macro")
        self.test_cm = MulticlassConfusionMatrix(num_classes=classes)

    def configure_optimizers(self) -> OptimizerLRScheduler:
        return torch.optim.AdamW([{"params": self.model.project.parameters(), "lr": self.hparams.lr}, ])

    @abstractmethod
    def extract_target(self, batch):
        pass

    def training_step(self, batch):
        # Labels
        y = self.extract_target(batch)
        x = dequantize(batch, self.default_dequantize_keys)
        pred = self.model(x)
        loss = self.criterion(pred, y)

        self.train_acc.update(pred, y)
        self.log("train_acc", self.train_acc, prog_bar=True, on_step=False, on_epoch=True)

        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
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


class AbstractEegAviClassificationTrainer(AbstractClassificationTrainer, ABC):
    def __init__(self, model: EegAviFineTune, seed: int, classes: int, lr=3e-4, backbone_lr=3e-5,
                 cbramod_lr=1e-5, weight_decay: float = 0.01):
        super().__init__(model, seed, lr, classes, backbone_lr)
        self.model: EegAviFineTune = model

    def configure_optimizers(self):
        project_parameters = [p for p in self.model.project.parameters() if p.requires_grad]
        project_ids = {id(p) for p in project_parameters}

        adapter: EegCbraModAdapter = self.model.get_pivot_adapter()

        cbramod_parameters = list(p for p in adapter.encoder.parameters() if p.requires_grad)
        cbra_params_ids = {id(p) for p in cbramod_parameters}

        eegavi_params = [p for p in self.model.encoder.parameters() if p.requires_grad and id(p) not in cbra_params_ids]
        eegavi_ids = {id(p) for p in eegavi_params}

        # No overlap
        assert project_ids.isdisjoint(cbra_params_ids)
        assert project_ids.isdisjoint(eegavi_ids)
        assert cbra_params_ids.isdisjoint(eegavi_ids)

        params = [{"params": project_parameters, "lr": self.hparams.lr}]
        if len(eegavi_params) != 0:
            params.append({"params": eegavi_params, "lr": self.hparams.backbone_lr})
        if len(cbramod_parameters) != 0:
            params.append({"params": cbramod_parameters, "lr": self.hparams.cbramod_lr})

        return torch.optim.AdamW(params, weight_decay=self.hparams.weight_decay, )


class AbstractCBraClassificationTrainer(AbstractClassificationTrainer, ABC):
    def __init__(self, model: CBraFineTune, seed: int, lr=3e-4, classes: int = 2,
                 backbone_lr=3e-5, weight_decay: float = 0.01):
        super().__init__(model, seed, lr, classes, backbone_lr)
        self.model: CBraFineTune = model

    def configure_optimizers(self) -> OptimizerLRScheduler:
        project_parameters = [p for p in self.model.project.parameters() if p.requires_grad]
        project_ids = {id(p) for p in project_parameters}

        cbramod_parameters = [p for p in self.model.encoder.parameters() if p.requires_grad]
        cbra_params_ids = {id(p) for p in cbramod_parameters}

        assert project_ids.isdisjoint(cbra_params_ids)
        params = [{"params": project_parameters, "lr": self.hparams.lr}]
        if len(cbramod_parameters) != 0:
            params.append({"params": cbramod_parameters, "lr": self.hparams.backbone_lr})

        return torch.optim.AdamW(params, weight_decay=self.hparams.weight_decay, )

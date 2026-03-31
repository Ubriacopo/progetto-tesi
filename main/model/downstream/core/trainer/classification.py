import lightning
import torch
from lightning.pytorch.utilities.types import OptimizerLRScheduler
from tensordict import TensorDict
from torch import nn

from main.core_data.media.audio import Audio
from main.core_data.media.eeg import EEG
from main.core_data.media.video import Video


def dequantize(x: dict | TensorDict, dequantize_keys: list[str], dtype=torch.float16):
    return_dict: dict = {}
    for key, td in x.items():
        if key in dequantize_keys and "data" in td:
            data = td["data"].to(dtype=dtype)
            data.mul_(td["scales"])
            td = {"data": data, "mask": td["mask"]}
        return_dict[key] = td
    return TensorDict.from_dict(return_dict, auto_batch_size=True)


class ClassificationTrainer(lightning.LightningModule):
    default_dequantize_keys = [EEG.modality_code(), Video.modality_code(), Audio.modality_code(), ]

    def __init__(self, model: nn.Module, labels: int, seed: int, lr=3e-4, backbone_lr=3e-5):
        super().__init__()
        self.model: nn.Module = model
        self.save_hyperparameters(ignore=["model"])

    def configure_optimizers(self) -> OptimizerLRScheduler:
        optimizer = torch.optim.AdamW([
            {"params": self.model.project.parameters(), "lr": self.hparams.lr},
          #  {"params": filter(lambda p: p.requires_grad, self.model.backbone.parameters()), "lr": self.hparams.backbone_lr},
        ])

        return optimizer

    def training_step(self, batch):
        # Labels
        y = batch["assessment", "score"][:, 0]
        x = dequantize(batch, self.default_dequantize_keys)
        pred = self.model(x)

        criterion = torch.nn.CrossEntropyLoss()
        loss = criterion(pred, y)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch):
        y = batch["assessment", "score"][:, 0]
        x = dequantize(batch, self.default_dequantize_keys)
        pred = self.model(x)

        criterion = torch.nn.CrossEntropyLoss()
        loss = criterion(pred, y)
        self.log("valid_loss", loss, prog_bar=True)
        # Estimated from the fact that classes are balanced except 1 that is one more numerous. From class distribution
        self.log("baseline", 0.1429, prog_bar=True)
        self.log("log(baseline)", 1.945, prog_bar=True)

        acc = (pred.argmax(dim=-1) == y).float().mean()
        self.log("valid_acc", acc, prog_bar=True)

        return loss

    def test_step(self, batch):
        y = batch["assessment", "score"][:, 0]
        x = dequantize(batch, self.default_dequantize_keys)
        pred = self.model(x)

        criterion = torch.nn.CrossEntropyLoss()
        loss = criterion(pred, y)
        self.log("test_loss", loss, prog_bar=True)
        self.log("log(baseline)", 1.945, prog_bar=True)

        acc = (pred.argmax(dim=-1) == y).float().mean()
        self.log("valid_acc", acc, prog_bar=True)
        return loss

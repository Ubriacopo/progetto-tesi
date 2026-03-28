import lightning
import tensordict
import torch
import torch.nn.functional as F
from jedi.inference.gradual.typing import TypedDict
from lightning.pytorch.utilities.types import OptimizerLRScheduler
from tensordict import TensorDict

from main.core_data.media.audio import Audio
from main.core_data.media.ecg import ECG
from main.core_data.media.eeg import EEG
from main.core_data.media.video import Video
from main.model.downstream.linear_probe import SimpleLinearProbe


class Scores(TypedDict):
    scores: torch.Tensor


class SupervisedInput(TypedDict):
    assessment: Scores


class SimpleLinearProbeTrainer(lightning.LightningModule):
    default_dequantize_keys = [EEG.modality_code(), Video.modality_code(), Audio.modality_code(), ECG.modality_code()]

    def __init__(self, probe: SimpleLinearProbe, dequantize_keys: list[str] = None, lr: float = 1e-3,
                 input_batch_size=(32, 10)):
        super().__init__()
        self.model = probe

        self.dequantize_keys: list[str] = dequantize_keys
        if self.dequantize_keys is None:
            self.dequantize_keys = self.default_dequantize_keys

        self.save_hyperparameters(ignore=["model"])

    def configure_optimizers(self) -> OptimizerLRScheduler:
        return {
            "optimizer": torch.optim.Adam(params=self.model.parameters(), lr=self.hparams.lr),
        }

    def dequantize(self, x: dict | TensorDict | SupervisedInput, dtype=torch.float16):
        return_dict = {}
        for key, td in x.items():
            if key in self.dequantize_keys:
                data = td["data"].to(dtype=dtype)
                data.mul_(td["scales"])

                td = {"data": data, "mask": td["mask"]}

            return_dict[key] = td
        return tensordict.from_dict(return_dict, batch_size=self.hparams.input_batch_size)

    def training_step(self, batch: SupervisedInput, batch_idx):
        y = batch["assessment", "scores"][:, 0].half()

        x = self.dequantize(batch)
        pred = self.model(x)

        loss = F.mse_loss(pred, y)

        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch: SupervisedInput, batch_idx):
        y = batch["assessment", "scores"][:, 0].half()
        x = self.dequantize(batch)
        pred = self.model(x)

        # 12-d loss
        loss = F.mse_loss(pred, y)
        self.log("val_loss", loss, prog_bar=True)

        # Core validation metrics with RMSE for interpretability and MAE robustness
        r_mse = torch.sqrt(loss)
        self.log("val_rmse", r_mse, prog_bar=True)
        mae = F.l1_loss(pred, y)
        self.log("val_mae", mae, prog_bar=True)

        # Pearson: linear agreement / trend alignment between prediction and target, independent of scale
        pearson_val = self.val_pearson(pred, y)  # returns (12,)
        self.log("val_pearson", pearson_val.mean())

        return loss

    def test_step(self, batch: SupervisedInput, batch_idx):
        y = batch["assessment", "scores"][:, 0].half()
        x = self.dequantize(batch)
        pred = self.model(x)

        # 12-d loss
        loss = F.mse_loss(pred, y)
        self.log("test_loss", loss, prog_bar=True)

        # Core validation metrics with RMSE for interpretability and MAE robustness
        r_mse = torch.sqrt(loss)
        self.log("test_rmse", r_mse, prog_bar=True)
        mae = F.l1_loss(pred, y)
        self.log("test_mae", mae, prog_bar=True)

        # Pearson: linear agreement / trend alignment between prediction and target, independent of scale
        pearson_val = self.test_pearson(pred, y)  # returns (12,)
        self.log("test_pearson", pearson_val.mean())

        return loss

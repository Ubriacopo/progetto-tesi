import lightning
import torch
import torch.nn.functional as F
from jedi.inference.gradual.typing import TypedDict
from lightning.pytorch.utilities.types import OptimizerLRScheduler
from tensordict import TensorDict
from torch import nn
from torchmetrics import PearsonCorrCoef

from main.core_data.media.audio import Audio
from main.core_data.media.ecg import ECG
from main.core_data.media.eeg import EEG
from main.core_data.media.video import Video


class Scores(TypedDict):
    scores: torch.Tensor


class SupervisedInput(TypedDict):
    assessment: Scores


class SimpleLinearProbeTrainer(lightning.LightningModule):
    default_dequantize_keys = [EEG.modality_code(), Video.modality_code(), Audio.modality_code(), ECG.modality_code()]

    def __init__(self, probe: nn.Module, labels: int, seed: int, dequantize_keys: list[str] = None, lr: float = 1e-4):
        super().__init__()
        self.model = probe

        self.val_pearson = PearsonCorrCoef(num_outputs=labels)
        self.test_pearson = PearsonCorrCoef(num_outputs=labels)

        self.dequantize_keys: list[str] = dequantize_keys
        if self.dequantize_keys is None:
            self.dequantize_keys = self.default_dequantize_keys

        # Mean predictor baseline, computed from train set in on_fit_start
        self.register_buffer("train_target_mean", torch.zeros(labels), persistent=True)
        self.save_hyperparameters(ignore=["probe", "train_target_mean"])

    def configure_optimizers(self) -> OptimizerLRScheduler:
        return {
            "optimizer": torch.optim.Adam(params=self.model.parameters(), lr=self.hparams.lr),
        }

    def dequantize(self, x: dict | TensorDict | SupervisedInput, dtype=torch.float32):
        return_dict: dict = {}
        for key, td in x.items():
            if key in self.dequantize_keys and "data" in td:
                data = td["data"].to(dtype=dtype)
                data.mul_(td["scales"])
                td = {"data": data, "mask": td["mask"]}
            return_dict[key] = td
        return TensorDict.from_dict(return_dict, auto_batch_size=True)

    def training_step(self, batch: SupervisedInput, batch_idx):
        y = self.extract_target(batch)

        x = self.dequantize(batch)
        pred = self.model(x)

        mask = torch.isfinite(y) & torch.isfinite(pred)
        #loss = ((pred - y) ** 2)[mask].mean()
        loss = F.mse_loss(pred, y)

        self.log("train_loss", loss, prog_bar=True)
        # Easier to understand how it works
        self.log("train_rmse", torch.sqrt(loss), prog_bar=True)

        return loss

    @staticmethod
    def extract_target(batch: SupervisedInput) -> torch.Tensor:
        # TODO In base ad applicazione
        y = batch["assessment", "scores"][:, 0].float()
        y = (y - 1) / 8.0

        return y

    @torch.no_grad()
    def on_fit_start(self):
        train_loader = self.trainer.datamodule.train_dataloader()

        total_sum = None
        total_count = 0

        was_training = self.training
        self.eval()
        for batch in train_loader:
            y = self.extract_target(batch).to(self.device)  # [B, 12]
            batch_sum = y.sum(dim=0)  # [12]
            if total_sum is None:
                total_sum = batch_sum
            else:
                total_sum += batch_sum

            total_count += y.shape[0]

        mean_y = total_sum / max(total_count, 1)  # [12]

        # We center on mean for better initialization
        self.train_target_mean.copy_(mean_y)
        nn.init.zeros_(self.model.project.weight)
        self.model.project.bias.copy_(mean_y)

        if was_training:
            self.train()
        self.print(f"Computed train mean baseline target: {mean_y}")

    def validation_step(self, batch: SupervisedInput, batch_idx):
        y = self.extract_target(batch)
        x = self.dequantize(batch)
        pred = self.model(x)

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

        # Mean-predictor baseline
        base_pred = self.train_target_mean.unsqueeze(0).expand_as(y)
        base_loss = F.mse_loss(base_pred, y)
        base_rmse = torch.sqrt(base_loss)
        base_mae = F.l1_loss(base_pred, y)
        self.log("val_baseline_rmse", base_rmse, prog_bar=True)
        self.log("val_baseline_mae", base_mae, prog_bar=True)

        rmse_per_dim = torch.sqrt(((pred - y) ** 2).mean(dim=0))
        for i, rmse_i in enumerate(rmse_per_dim):
            self.log(f"val_rmse_dim_{i}", rmse_i, prog_bar=False)

        r2 = 1 - loss / base_loss
        self.log("val_r2", r2, prog_bar=True)
        return loss

    def test_step(self, batch: SupervisedInput, batch_idx):
        y = self.extract_target(batch)

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

import lightning
import torch
import torch.nn.functional as F
from lightning.pytorch.utilities.types import OptimizerLRScheduler, STEP_OUTPUT
from torchmetrics.regression import PearsonCorrCoef

from main.model.downstream.faced.model import FacedLinearProbe, FacedInput


class FacedSupervisedInput(FacedInput):
    labels: torch.Tensor  # [b, 12] (12 labels)


# todo data dequantiazion in loader and not in train step?
class FacedProbeTrainer(lightning.LightningModule):
    def __init__(self, model: FacedLinearProbe, lr: float = 1e-4):
        super().__init__()
        self.model = model

        self.val_pearson = PearsonCorrCoef(num_outputs=12)
        self.test_pearson = PearsonCorrCoef(num_outputs=12)
        self.save_hyperparameters(ignore=["model"])

    def configure_optimizers(self) -> OptimizerLRScheduler:
        return dict(
            optimizer=torch.optim.Adam(
                params=self.model.parameters(),
                lr=self.hparams.lr
            )
        )

    def training_step(self, batch: FacedSupervisedInput, batch_idx):
        y = batch["labels"]
        pred = self.model(batch)

        # 12-d loss
        loss = F.mse_loss(pred, y)
        self.log("train_loss", loss, prog_bar=True)

        return loss

    def validation_step(self, batch: FacedSupervisedInput, batch_idx) -> STEP_OUTPUT:
        y = batch["labels"]
        pred = self.model(batch)

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

    def test_step(self, batch: FacedSupervisedInput, batch_idx):
        y = batch["labels"]
        pred = self.model(batch)

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

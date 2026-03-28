import lightning
import tensordict
import torch
import torch.nn.functional as F
from lightning.pytorch.utilities.types import OptimizerLRScheduler, STEP_OUTPUT
from torchmetrics.regression import PearsonCorrCoef

from main.core_data.media.audio import Audio
from main.core_data.media.ecg import ECG
from main.core_data.media.eeg import EEG
from main.core_data.media.video import Video
from main.model.downstream.mioamicogesu.model import FusionLinearProbe


class FacedSupervisedInput:
    assessment: torch.Tensor  # [b, 12] (12 labels)


# todo data dequantiazion in loader and not in train step?
class FusionProbeTrainer(lightning.LightningModule):
    def __init__(self, model: FusionLinearProbe, lr: float = 1e-3):
        super().__init__()
        self.model = model

        self.val_pearson = PearsonCorrCoef(num_outputs=5)
        self.test_pearson = PearsonCorrCoef(num_outputs=5)
        self.dequantize_keys = [
            EEG.modality_code(),
            Video.modality_code(),
            Audio.modality_code(),
            ECG.modality_code(),
        ]
        self.save_hyperparameters(ignore=["model"])

    def configure_optimizers(self) -> OptimizerLRScheduler:
        return dict(
            optimizer=torch.optim.Adam(
                params=self.model.parameters(),
                lr=self.hparams.lr
            )
        )

    def dequantize(self, x: FacedSupervisedInput, dtype=torch.float16):
        output: FacedSupervisedInput = {}
        for key, td in x.items():
            if key in self.dequantize_keys:
                data = td["data"].to(dtype=dtype, non_blocking=True)
                data.mul_(td["scales"])  # For optimization reasons (I dislike it)
                td = {"data": data, "mask": td["mask"]}
            output[key] = td
            # todo pass
        return tensordict.from_dict(output, batch_size=[x["eeg"]["data"].shape[0], x["eeg"]["data"].shape[1]])

    def training_step(self, batch: FacedSupervisedInput, batch_idx):
        y = batch["assessment"]["scores"][:, 0].half()

        x = self.dequantize(batch)
        pred = self.model(x)
        # pred = [self.model(x) for i in x]

        # 12-d loss
        loss = F.mse_loss(pred, y)
        self.log("train_loss", loss, prog_bar=True)

        return loss

    def validation_step(self, batch: FacedSupervisedInput, batch_idx) -> STEP_OUTPUT:
        y = batch["assessment"]["scores"][:, 0].half()
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

    def test_step(self, batch: FacedSupervisedInput, batch_idx):
        y = batch["assessment"]["scores"][:, 0].half()
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
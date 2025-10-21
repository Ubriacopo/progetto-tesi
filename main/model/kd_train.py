from typing import Literal

import lightning as L
import torch.optim
from pytorch_lightning.utilities.types import OptimizerLRScheduler
from torchmetrics.functional import concordance_corrcoef

from main.core_data.media.assessment.assessment import Assessment
from main.model.EEGAVI.base_model import WeaklySupervisedEegBaseModel, WeaklySupervisedEegBaseModelOutputs
from main.model.VATE.constrastive_model import MaskedContrastiveModel, MaskedContrastiveModelOutputs
from main.model.loss import masked_info_nce_2d, masked_cosine_similarity, SiglipLoss, masked_mse_kd, masked_cosine_kd
from main.utils.data import MaskedValue


class EegAviKdVateMaskedModule(L.LightningModule):
    pass


class EegAviKdVateMaskedSemiSupervisedModule(L.LightningModule):
    def __init__(self,
                 student: WeaklySupervisedEegBaseModel,
                 teacher: MaskedContrastiveModel,
                 kd_loss_weight: float,
                 fusion_loss_weight: float,
                 weakly_supervised_weight: float,
                 ecg_correction_weight: float,
                 lr: float = 1e-4,
                 kd_temperature: float = .2):
        super().__init__()
        self.student = student
        self.teacher = teacher

        self.verbose = True
        self.siglip_loss = SiglipLoss(init_tau=0.07, init_bias=-10, stop_grad_target=False, verbose=self.verbose)
        self.aud_siglip_loss = SiglipLoss(init_tau=0.07, init_bias=-10, stop_grad_target=False, verbose=self.verbose)

        # Weight terms for loss parts
        self.alpha: float = kd_loss_weight
        self.beta: float = fusion_loss_weight
        self.gamma: float = weakly_supervised_weight
        self.theta: float = ecg_correction_weight
        self.kd_temperature = kd_temperature
        self.lr = lr

    def configure_optimizers(self) -> OptimizerLRScheduler:
        # todo warmup lr
        return torch.optim.Adam(weight_decay=0.01, params=[
            {"params": self.student.parameters(), "lr": self.lr},
            {"params": self.siglip_loss.parameters(), "lr": self.lr * 10, "weight_decay": 0.0},
            {"params": self.aud_siglip_loss.parameters(), "lr": self.lr * 10, "weight_decay": 0.0}
        ])


    def compute_kd_loss(self, student_out: dict[str, MaskedValue], teacher_out: MaskedContrastiveModelOutputs):
        total_loss = 0.0
        total_samples = 0
        key: Literal['vid', 'txt', 'aud']
        for key in teacher_out.keys():
            if key not in student_out:
                continue

            loss_k, n_rows = masked_cosine_kd(
                za=student_out[key]["data"],
                za_mask=student_out[key]["mask"],
                zb=teacher_out[key]["data"],
                zb_mask=teacher_out[key]["mask"],
                # tau=self.kd_temperature
            )

            if n_rows > 0:
                total_loss += loss_k * n_rows  # Weighted by valid samples
                total_samples += n_rows

        # Simple average
        loss = total_loss / max(total_samples, 1)
        self.log("kd_loss", loss, on_epoch=True, on_step=True, prog_bar=True)
        return loss * self.alpha

    # todo sistemare
    def compute_fusion_loss(self, fused_output: torch.Tensor, modality_outputs: dict[str, MaskedValue]):
        base_loss = torch.tensor(0.0, device=fused_output.device)
        for key, value in modality_outputs.items():
            print(f"\nFor key {key}:")
            # before is 3D. We mean over valid masked rows
            y_before = value["data"]
            mask_before = value["mask"].unsqueeze(-1)
            y_before = (y_before * mask_before).sum(dim=1) / mask_before.sum(dim=1)
            if key == "aud":
                l = self.aud_siglip_loss(fused_output, y_before)
            else:
                l = self.siglip_loss(fused_output, y_before)
            self.log("siglip_" + key, l, on_step=True, on_epoch=True, prog_bar=True)
            base_loss += l

        return base_loss

    def compute_supervised_loss(self, pred: torch.Tensor, target: torch.Tensor):
        # Compute concordance correlation coefficient that measures the agreement between two variables.
        # In emotion regression (valence, arousal, dominance), this is the standard metric and loss used in benchmarks. TODO verify
        # Correlation and agreement rather than absolute distance
        tol = 1e-8

        pred = (pred - pred.mean(dim=0)) / (pred.std(dim=0) + 1e-8)
        target = (target - target.mean(dim=0)) / (target.std(dim=0) + 1e-8)

        target_std = target.std(dim=0)
        ccc = concordance_corrcoef(pred[:, target_std > tol], target[:, target_std > tol]).mean()
        self.log("supervised", ccc, on_epoch=True, prog_bar=True)
        return (1 - ccc) * self.gamma

    def training_step(self, batch, batch_idx):
        stud_out: WeaklySupervisedEegBaseModelOutputs = self.student(batch["student"], use_kd=True)
        with torch.inference_mode():
            teacher_out: MaskedContrastiveModelOutputs = self.teacher(**batch["teacher"])

        loss = (
            self.compute_kd_loss(student_out=stud_out.kd_outs, teacher_out=teacher_out)
            # self.compute_fusion_loss(fused_output=stud_out.embeddings, modality_outputs=stud_out.multimodal_outs)
            # self.compute_supervised_loss(pred=stud_out.pred, target=batch["student"][Assessment.modality_code()])
        )

        # Optional term
        if False and "ecg" in stud_out.multimodal_outs:  # We have ECG
            loss = loss + self.theta * masked_cosine_similarity(
                za=stud_out.multimodal_outs["eeg"]["data"],
                # ECG goes through the Perceiver Resampler so it is 4D
                zb=stud_out.multimodal_outs["ecg"]["data"].mean(dim=-2),
                present=stud_out.multimodal_outs["eeg"]["mask"] & stud_out.multimodal_outs["ecg"]["mask"].any(dim=-1)
            )

        self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True)

        return loss

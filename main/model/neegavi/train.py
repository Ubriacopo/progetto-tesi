from typing import Literal

import lightning as L
import torch
import torch.nn.functional as F
from lightning.pytorch.utilities.types import OptimizerLRScheduler
from torch import nn
from torchmetrics.functional import pearson_corrcoef, concordance_corrcoef

from main.core_data.media.assessment.assessment import Assessment
from main.model.VATE.constrastive_model import MaskedContrastiveModel, MaskedContrastiveModelOutputs
from main.model.loss import SiglipLoss
from main.model.neegavi.base_model import WeaklySupervisedNEEEGBaseModel
from main.model.neegavi.utils import WeaklySupervisedEegBaseModelOutputs
from main.utils.data import MaskedValue


class EegAviKdVateMaskedSemiSupervisedModule(L.LightningModule):
    def __init__(
            self,
            student: WeaklySupervisedNEEEGBaseModel, teacher: MaskedContrastiveModel,
            kd_loss_weight: float, fusion_loss_weight: float, weakly_supervised_weight: float,
            fusion_metrics: list[str], lr: float, kd_temperature: float
    ):
        super().__init__()

        self.verbose: bool = False

        self.student = student
        self.teacher = teacher

        self.siglip_losses: nn.ModuleDict = nn.ModuleDict()
        for fusion_metric in fusion_metrics:
            loss_fn = SiglipLoss(init_tau=0.07, init_bias=-10, stop_grad_target=False, verbose=self.verbose)
            self.siglip_losses.add_module(fusion_metric, loss_fn)

        # Just to debug atm
        self.use_kd_loss = False
        self.use_fusion_loss = False
        self.use_supervised_loss = True

        # Hyperparameters
        self.lr: float = lr
        self.kd_temperature: float = kd_temperature
        # Weights of different losses combined
        self.alpha: float = kd_loss_weight
        self.beta: float = fusion_loss_weight
        self.gamma: float = weakly_supervised_weight

    def configure_optimizers(self) -> OptimizerLRScheduler:
        siglip_common_optim_configs = [
            {"params": i.parameters(), "lr": self.lr * 10, "weight_decay": 0.0}
            for i in self.siglip_losses.values()
        ]
        student_params = [{"params": self.student.parameters(), "lr": self.lr}]
        return torch.optim.Adam(weight_decay=.01, params=siglip_common_optim_configs + student_params)

    def training_step(self, batch, batch_idx):
        stud_out: WeaklySupervisedEegBaseModelOutputs = self.student(batch["student"], use_kd=True)
        with torch.inference_mode():
            teacher_out: MaskedContrastiveModelOutputs = self.teacher(**batch["teacher"])

        loss = .0

        if self.use_kd_loss:
            loss = loss + self.compute_kd_loss(student_out=stud_out.kd_outs, teacher_out=teacher_out) * self.alpha

        if self.use_fusion_loss:
            loss = loss + self.compute_fusion_loss(stud_out.embeddings, stud_out.multimodal_outs) * self.beta

        if self.use_supervised_loss:
            targets = batch["student"][Assessment.modality_code()].float()
            loss = loss + self.compute_supervised_loss(pred=stud_out.pred, target=targets) * self.gamma

        self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    @staticmethod
    @torch.no_grad()
    def siglip_random_baseline(loss_fn, a, b):
        # shuffle targets to break alignment
        idx = torch.randperm(b.shape[0], device=b.device)
        return loss_fn(a, b[idx])

    def compute_kd_loss(self, student_out: dict[str, MaskedValue], teacher_out: MaskedContrastiveModelOutputs) -> float:
        loss = .0

        key: Literal['vid', 'txt', 'aud']
        for key in teacher_out.keys():
            if key not in student_out:
                continue  # This element is not KD or is absent from teacher so we cannot learn from it
            loss_fn = SiglipLoss(init_tau=0.05, init_bias=-10, stop_grad_target=True)
            loss_fn.to('cuda')
            # modality_loss = self.siglip_losses[key](student_out[key]["data"], teacher_out[key]['data'])
            modality_loss = loss_fn(student_out[key]["data"], teacher_out[key]['data'])
            self.log(
                f"kd_rand_{key}",
                self.siglip_random_baseline(loss_fn, student_out[key]["data"], teacher_out[key]['data'], ),
                on_epoch=True, on_step=False, prog_bar=True
            )

            self.log(f"kd_loss_{key}", modality_loss, on_epoch=True, on_step=False, prog_bar=True)
            loss = loss + modality_loss

        self.log("kd_loss", loss, on_epoch=True, on_step=False, prog_bar=True)
        return loss

    def compute_fusion_loss(self, fused_output: torch.Tensor, modality_outputs: dict[str, MaskedValue]) -> torch.Tensor:
        base_loss = torch.tensor(0.0, device=fused_output.device)
        for key, value in modality_outputs.items():
            self.verbose and print(f"\nFor key {key}:")
            y_before, mask_before = value["data"], value["mask"].unsqueeze(-1)

            y_before = (y_before * mask_before).sum(dim=1) / mask_before.sum(dim=1)
            mod_loss = self.siglip_losses[key](fused_output, y_before)
            self.log("siglip_" + key, mod_loss, on_epoch=True, on_step=False, prog_bar=True)
            base_loss = base_loss + mod_loss

        # Cumulative loss between all modalities non normalized
        return base_loss

    def compute_supervised_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Compute concordance correlation coefficient that measures the agreement between two variables.
        # In emotion regression (valence, arousal, dominance), this is the standard metric and loss used in benchmarks.
        #
        # Correlation and agreement rather than absolute distance.

        tol = 1e-8

        T = 2  # warmup length (epochs) – tune this

        t_std = target.std(dim=0, unbiased=False)
        p_std = pred.std(dim=0, unbiased=False)
        mask = (t_std > tol) & (p_std > tol)

        if mask.any():
            pred, target = pred[:, mask].float(), target[:, mask].float()
            pearson = pearson_corrcoef(pred, target).mean().float()
            concordance = concordance_corrcoef(pred, target).mean().float()
            w = min(1.0, float(self.current_epoch) / T)  # or cosine ramp
            one = pred.new_tensor(1.0)  # ensures dtype/device match

            loss = (1 - w) * (one - pearson) + w * (one - concordance)
            self.log("supervised (CCC & Pearson)", loss, on_epoch=True, on_step=False, prog_bar=True)
            loss = loss.to(pred.dtype)
            return loss
        else:
            loss = F.mse_loss(pred, target).float()
            self.log("supervised", loss, on_epoch=True, on_step=False, prog_bar=True)
            return loss

from typing import Literal

import lightning as L
import torch.nn.functional as F
import torch.optim
from pytorch_lightning.utilities.types import OptimizerLRScheduler
from torchmetrics import ConcordanceCorrCoef
from torchmetrics.functional import concordance_corrcoef

from main.core_data.media.assessment.assessment import Assessment
from main.model.EEGAVI.EEGAVI import WeaklySupervisedEEGAVI, WeaklySupervisedEEGAVIOutputs
from main.model.VATE.constrastive_model import MaskedContrastiveModel, MaskedContrastiveModelOutputs
from main.model.loss import masked_info_nce_2d, siglip, masked_cosine_similarity, diagnose_siglip, SiglipLoss
from main.utils.data import MaskedValue


class EegAviKdVateMaskedModule(L.LightningModule):
    pass


class EegAviKdVateMaskedSemiSupervisedModule(L.LightningModule):
    def __init__(self,
                 student: WeaklySupervisedEEGAVI,
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
        # TODO Find defaults

        self.siglip_loss = SiglipLoss(init_tau=0.2, stop_grad_target=True)

        # Weight terms for loss parts
        self.alpha: float = kd_loss_weight
        self.beta: float = fusion_loss_weight
        self.gamma: float = weakly_supervised_weight
        self.theta: float = ecg_correction_weight
        self.kd_temperature = kd_temperature
        self.lr = lr

    def configure_optimizers(self) -> OptimizerLRScheduler:
        # todo warmup lr
        return torch.optim.Adam([
            {"params": self.student.parameters()},
            {"params": self.siglip_loss.parameters()}
        ], lr=self.lr, weight_decay=0
        )

    def compute_kd_loss(self, student_out: dict[str, MaskedValue], teacher_out: MaskedContrastiveModelOutputs) \
            -> float | torch.Tensor:
        numerator, denominator = .0, 0
        key: Literal['vid', 'txt', 'aud']
        for key in teacher_out.keys():
            if key not in student_out:
                continue  # Keys do never match so no kd on this.
            loss_k, n_rows = masked_info_nce_2d(
                za=student_out[key]["data"], za_mask=student_out[key]["mask"],
                zb=teacher_out[key]["data"], zb_mask=teacher_out[key]["mask"],
                tau=self.kd_temperature
            )

            loss_k /= torch.clamp(torch.log(torch.tensor(n_rows, device=loss_k.device, dtype=loss_k.dtype)), min=1e-6)
            # todo possibile problema potrebbe essere dovuto a batch size piccola:
            # Option 1: Gradient Accumulation (Recommended) Simulate larger batches without memory increase:
            #       Memory Bank alternativa piu robusta (Al momento siamo in un punto "good enough" per batch che abbiamo)
            #       Alternativa ancora sarebbe quella di usare sigliploss direttamente che dipende meno da bathc size?
            numerator += loss_k * n_rows  # Weighed by number of valid rows
            denominator += n_rows  # Track total number of valid rows

        loss = numerator / max(denominator, 1)
        self.log("kd_loss", loss, on_epoch=True)
        return loss * self.alpha

    def compute_fusion_loss(self, fused_output: torch.Tensor, modality_outputs: dict[str, MaskedValue]):
        numerator = torch.zeros((), device=fused_output.device, dtype=fused_output.dtype)
        denominator = torch.zeros((), device=fused_output.device, dtype=fused_output.dtype)
        # TODO: Broken fusion module somehow.
        for key, value in modality_outputs.items():
            z, mask = value["data"], value["mask"]

            if mask.dim() == 3:
                mask = mask[:, :, 0].squeeze(dim=-1)
            # Usually stud_out_multimodal out: (b, T, P, D) (Differs for EEG)
            # Patch mean: (Has no mask patches are always max rank).
            if z.dim() > 3:
                # We have patch tokens so we mean over them
                z = z.mean(dim=-2)

            w = mask.float().sum(dim=-1, keepdim=True).clamp_min(1e-6)  # Normalization factor
            z = (z * mask.float().unsqueeze(-1)).sum(dim=-2) / w

            z_norms = z.norm(dim=-1)
            valid = (mask.any(dim=1)) & (z_norms > 1e-6)
            if valid.sum() == 0:
                continue

            n = valid.sum()
            modality_loss = self.siglip_loss(fused_output[valid], z[valid])

            numerator += modality_loss * n  #
            denominator += n ** 2  # Count the samples that have to do with loss calc

        loss = numerator / denominator.clamp(min=1e-6)
        self.log("unweighted_fusion_loss", loss, on_epoch=True)
        return loss * self.beta

    def compute_supervised_loss(self, pred: torch.Tensor, target: torch.Tensor):
        # Compute concordance correlation coefficient that measures the agreement between two variables.
        # In emotion regression (valence, arousal, dominance), this is the standard metric and loss used in benchmarks. TODO verify
        # Correlation and agreement rather than absolute distance
        tol = 1e-8
        target_std = target.std(dim=0)
        ccc = concordance_corrcoef(pred[:, target_std > tol], target[:, target_std > tol]).mean()
        self.log("unweighted_supervised_loss", ccc, on_epoch=True)
        return (1 - ccc) * self.gamma

    def training_step(self, batch, batch_idx):
        stud_out: WeaklySupervisedEEGAVIOutputs = self.student(batch["student"], use_kd=True)
        with torch.inference_mode():
            teacher_out: MaskedContrastiveModelOutputs = self.teacher(**batch["teacher"])

        loss = (
            # self.compute_kd_loss(student_out=stud_out.kd_outs, teacher_out=teacher_out)
            self.compute_fusion_loss(fused_output=stud_out.embeddings, modality_outputs=stud_out.multimodal_outs)
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

        self.log("train_loss", loss, prog_bar=True, on_epoch=True)

        return loss

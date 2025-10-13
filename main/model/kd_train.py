from typing import Literal

import lightning as L
import torch.nn.functional as F
import torch.optim
from pytorch_lightning.utilities.types import OptimizerLRScheduler
from torchmetrics import ConcordanceCorrCoef

from main.core_data.media.assessment.assessment import Assessment
from main.model.EEGAVI.EEGAVI import WeaklySupervisedEEGAVI, WeaklySupervisedEEGAVIOutputs
from main.model.VATE.constrastive_model import MaskedContrastiveModel, MaskedContrastiveModelOutputs
from main.model.loss import masked_info_nce_2d, siglip, masked_cosine_similarity, diagnose_siglip
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
        # Weight terms for loss parts
        self.alpha: float = kd_loss_weight
        self.beta: float = fusion_loss_weight
        self.gamma: float = weakly_supervised_weight
        self.theta: float = ecg_correction_weight
        self.kd_temperature = kd_temperature
        self.lr = lr

    def configure_optimizers(self) -> OptimizerLRScheduler:
        # todo warmup lr
        return torch.optim.AdamW(self.student.parameters(), lr=self.lr)

    def compute_kd_loss(self, student_out: dict[str, MaskedValue], teacher_out: MaskedContrastiveModelOutputs) \
            -> float | torch.Tensor:
        loss, w = .0, 0
        key: Literal['vid', 'txt', 'aud']
        for key in teacher_out.keys():
            if key not in student_out:
                continue  # Keys do never match so no kd on this.
            loss_k, n_rows = masked_info_nce_2d(
                za=student_out[key]["data"], za_mask=student_out[key]["mask"],
                zb=teacher_out[key]["data"], zb_mask=teacher_out[key]["mask"],
                tau=self.kd_temperature
            )
            # todo possibile problema potrebbe essere dovuto a batch size piccola:
            # Option 1: Gradient Accumulation (Recommended) Simulate larger batches without memory increase:
            #       Memory Bank alternativa piu robusta (Al momento siamo in un punto "good enough" per batch che abbiamo)
            loss += loss_k * n_rows  # Weighed by number of valid rows
            w += n_rows  # Track total number of valid rows

        loss = (loss / w) if w != 0 else student_out[next(iter(student_out.keys()))]["data"].new_zeros()
        self.log("kd_loss", loss, on_epoch=True)
        return loss * self.alpha

    # TODO ma va mascherato? Non credo da come ho fatto pipe di EEG ma controlla
    def compute_fusion_loss(self, fused_output: torch.Tensor, modality_outputs: dict[str, MaskedValue]):
        loss = .0
        present_modalities = 0
        # todo revisiona ci sono errorini
        # todo ma giusto cosi anche se maschero a riga? metti che modlaita per sample x manca?

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

            z_valid = z[valid]
            z_valid = F.normalize(z_valid, p=2, dim=-1)
            modality_loss = diagnose_siglip(fused_output[valid], z_valid)

            loss += modality_loss
            present_modalities += 1

        # Normalize by the correct number of modalities used
        loss = loss / max(present_modalities, 1)
        self.log("unweighted_fusion_loss", loss, on_epoch=True)
        return loss * self.beta

    def compute_supervised_loss(self, pred: torch.Tensor, target: torch.Tensor):
        # Compute concordance correlation coefficient that measures the agreement between two variables.
        # In emotion regression (valence, arousal, dominance), this is the standard metric and loss used in benchmarks. TODO verify
        # Correlation and agreement rather than absolute distance
        ccc = ConcordanceCorrCoef(num_outputs=pred.shape[1]).to(pred.device)(pred, target).mean()
        self.log("unweighted_supervised_loss", ccc, on_epoch=True)
        return (1 - ccc) * self.gamma

    def training_step(self, batch, batch_idx):
        stud_out: WeaklySupervisedEEGAVIOutputs = self.student(batch["student"], use_kd=True)
        with torch.inference_mode():
            teacher_out: MaskedContrastiveModelOutputs = self.teacher(**batch["teacher"])

        loss = (
                self.compute_kd_loss(student_out=stud_out.kd_outs, teacher_out=teacher_out)
                + self.compute_fusion_loss(fused_output=stud_out.embeddings, modality_outputs=stud_out.multimodal_outs)
                + self.compute_supervised_loss(pred=stud_out.pred, target=batch["student"][Assessment.modality_code()])
        )

        # Optional term
        if "ecg" in stud_out.multimodal_outs:  # We have ECG
            loss = loss + self.theta * masked_cosine_similarity(
                za=stud_out.multimodal_outs["eeg"]["data"],
                # ECG goes through the Perceiver Resampler so it is 4D
                zb=stud_out.multimodal_outs["ecg"]["data"].mean(dim=-2),
                present=stud_out.multimodal_outs["eeg"]["mask"] & stud_out.multimodal_outs["ecg"]["mask"].any(dim=-1)
            )

        self.log("train_loss", loss, prog_bar=True, on_epoch=True)

        return loss

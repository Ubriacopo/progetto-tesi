import lightning as pl
import torch
import torch.nn.functional as F
from lightning.pytorch.utilities import CombinedLoader
from lightning.pytorch.utilities.types import STEP_OUTPUT
from torch.utils.data import DataLoader

from core_data.dataset import FlexibleEmbeddingsSpecMediaDataset
from model.EEGAVI.EEGAVI import EEGAVI, EEGAVIOutputs
from model.EEGAVI.interleaved_EEGAVI.interleaved_model import get_interleaved_EEG_AVI
from model.VATE.constrastive_model import ContrastiveModel, MaskedContrastiveModel, MaskedContrastiveModelOutputs
from model.loss import siglip
from model.utils import MaskedResult


class EegAviKdVateMaskedModule(pl.LightningModule):
    def __init__(self, student: EEGAVI, teacher: MaskedContrastiveModel):
        super().__init__()
        self.student: EEGAVI = student
        self.teacher: MaskedContrastiveModel = teacher

    def measure_modality_kd_loss(self, teacher: MaskedResult, student: MaskedResult) -> torch.Tensor:
        # todo vedi teacher mask shape
        idx = (student["mask"].any(dim=1) & teacher["mask"].any(dim=0)).nonzero(as_tuple=True)[0]
        if idx.numel() == 0:
            return torch.tensor(.0)
        # InfoNCE loss.
        # Normalize so that dot product similarity == to cosine similarity.
        s = F.normalize(student["data"][idx], dim=-1)
        t = F.normalize(teacher["data"][idx], dim=-1)
        # Take what is valid for both
        logits = (s @ t.T) / self.tau
        return F.cross_entropy(logits, torch.arange(idx.numel(), device=s.device))

    def training_step(self, batch: dict[str, dict], batch_idx) -> STEP_OUTPUT:
        x_teacher, x_student = batch["kd"], batch["sup"]
        with torch.inference_mode():
            teacher_out: MaskedContrastiveModelOutputs = self.teacher(**x_teacher)
        stud_out: EEGAVIOutputs = self.student(x_student, use_kd=True)

        # KD Loss
        kd_loss = .0
        for key in teacher_out.keys():
            if key not in stud_out.kd_outs:
                continue  # Keys do never match so no kd on this.
            kd_loss += self.measure_modality_kd_loss(teacher_out[key], stud_out.kd_outs[key])

        # Fusion Loss
        fusion_loss = .0
        present_modalities: int = 0
        for key, value in stud_out.multimodal_outs.items():
            z, mask = value["data"], value["mask"]
            if mask.dim() == 3:
                mask = mask[:, :, 0].squeeze()
            # stud_out_multimodal out (b, T, P, D)
            # Patch mean: (Has no mask patches are always max rank).
            if z.dim() > 3:
                # We have patch tokens so we mean over them. (In case they exist).
                z = z.mean(dim=-2)

            # Timed masked mean:
            w = mask.float().sum(dim=-1, keepdim=True).clamp_min(1e-6)  # Normalization factor
            z = (z * mask.float().unsqueeze(-1)).sum(dim=-2) / w
            z = F.normalize(z, dim=-1)

            valid = mask.any(dim=1)

            if valid.any():
                # stud.out.embeddings: (b, D) already. No need for further transformations.
                fusion_loss += siglip(stud_out.embeddings[valid], z[valid])
                present_modalities += 1

        l_cons = .0
        if "ecg" in stud_out.multimodal_outs:  # We have ECG
            z_eeg = stud_out.multimodal_outs["eeg"]["data"]
            z_ecg = stud_out.multimodal_outs["ecg"]["data"]
            present = stud_out.multimodal_outs["eeg"]["mask"] & stud_out.multimodal_outs["ecg"]["mask"].any(dim=-1)
            w = present.float()
            # ECG goes through the Perceiver Resampler so it is 4D
            z_ecg = z_ecg.mean(dim=-2)
            # masked mean of (1 - cos)
            l_cons = ((1 - F.cosine_similarity(z_eeg, z_ecg, dim=-1)) * w).sum() / w.sum().clamp_min(1.0)

        loss = self.alpha * kd_loss + self.beta * (fusion_loss / max(present_modalities, 1)) + l_cons * self.gamma
        self.log("train_loss", loss)

        return loss


class EegAviKdModule(pl.LightningModule):
    def __init__(self, student: EEGAVI, teacher: ContrastiveModel,
                 alpha: float = 1.0, beta: float = 1., gamma: float = 0.5, lr: float = 1e-4):
        super().__init__()
        self.student = student
        self.teacher = teacher

        for p in self.teacher.parameters():
            p.requires_grad = False

        # Weight terms for loss parts
        self.alpha: float = alpha
        self.beta: float = beta
        self.gamma: float = gamma

        self.lr = lr
        self.tau = 0.2  # Temperature

    def measure_modality_kd_loss(self, teacher_x: torch.Tensor, teacher_mask: torch.Tensor,
                                 student_x: torch.Tensor, student_mask: torch.Tensor) -> torch.Tensor:
        idx = (student_mask.any(dim=1) & teacher_mask.any(dim=1)).nonzero(as_tuple=True)[0]
        if idx.numel() == 0:
            return torch.tensor(.0)
        # InfoNCE loss.
        # Normalize so that dot product similarity == to cosine similarity.
        s = F.normalize(student_x[idx], dim=-1)
        t = F.normalize(teacher_x[idx], dim=-1)
        # Take what is valid for both. todo questi step di computazione maschera da fare in modello e non qui
        logits = (s @ t.T) / self.tau
        return F.cross_entropy(logits, torch.arange(idx.numel(), device=s.device))

    def training_step(self, batch: dict[str, dict], batch_idx) -> STEP_OUTPUT:
        """
        TODO: improvmeents: Potrei aggiungere altro alla loss ma parto con questa
        :param batch:
        :param batch_idx:
        :return:
        """
        x_teacher = batch["kd"]
        x_student = batch["sup"]

        with torch.inference_mode():
            # Get Knowledge Distillation terms
            kd_vid, kd_aud, kd_txt, _ = self.teacher(
                x_teacher["vid"] if "vid" in x_teacher else None,
                x_teacher["aud"] if "aud" in x_teacher else None,
                x_teacher["txt"] if "txt" in x_teacher else None
            )

        stud_out: EEGAVIOutputs = self.student(x_student, use_kd=True)

        kd_loss = .0

        if "vid" in stud_out.kd_outs:
            vid_mask = stud_out.kd_outs["vid"]["mask"]
            kd_loss += self.measure_modality_kd_loss(
                # Potrei anche non farmi dare la maschera dal teacher. Tanto se non ha un elemento lo student anche
                # il teacher non lo ha visto che sono allineati i dataset.
                teacher_x=kd_vid, teacher_mask=vid_mask,
                student_x=stud_out.kd_outs["vid"]["data"], student_mask=vid_mask
            )

        if "aud" in stud_out.kd_outs:
            aud_mask = stud_out.kd_outs["aud"]["mask"]
            kd_loss += self.measure_modality_kd_loss(
                teacher_x=kd_aud, teacher_mask=aud_mask,
                student_x=stud_out.kd_outs["aud"]["data"], student_mask=aud_mask
            )

        if "txt" in stud_out.kd_outs:
            txt_mask = stud_out.kd_outs["txt"]["mask"]
            kd_loss += self.measure_modality_kd_loss(
                teacher_x=kd_txt, teacher_mask=txt_mask,
                student_x=stud_out.kd_outs["txt"]["data"], student_mask=txt_mask
            )

        # Apply SigLipLoss now
        fusion_loss = .0
        present_modalities: int = 0
        for key, value in stud_out.multimodal_outs.items():
            # embeddings shape: (b, T, D) ->
            z, mask = value["data"], value["mask"]
            if mask.dim() == 3:
                mask = mask[:, :, 0].squeeze()
            # stud_out_multimodal out (b, T, P, D)
            # Patch mean: (Has no mask patches are always max rank).
            if z.dim() > 3:
                # We have patch tokens so we mean over them. (In case they exist).
                z = z.mean(dim=-2)
            # Timed masked mean:
            w = mask.float().sum(dim=-1, keepdim=True).clamp_min(1e-6)  # Normalization factor
            z = (z * mask.float().unsqueeze(-1)).sum(dim=-2) / w
            z = F.normalize(z, dim=-1)
            valid = mask.any(dim=1)
            if valid.any():
                # stud.out.embeddings: (b, D) already. No need for further transformations.
                fusion_loss += siglip(stud_out.embeddings[valid], z[valid])
                present_modalities += 1

        l_cons = .0
        if "ecg" in stud_out.multimodal_outs:  # We have ECG
            z_eeg = stud_out.multimodal_outs["eeg"]["data"]
            z_ecg = stud_out.multimodal_outs["ecg"]["data"]
            present = stud_out.multimodal_outs["eeg"]["mask"] & stud_out.multimodal_outs["ecg"]["mask"].any(dim=-1)
            w = present.float()
            # ECG goes through the Perceiver Resampler so it is 4D
            z_ecg = z_ecg.mean(dim=-2)
            # masked mean of (1 - cos)
            l_cons = ((1 - F.cosine_similarity(z_eeg, z_ecg, dim=-1)) * w).sum() / w.sum().clamp_min(1.0)

        loss = self.alpha * kd_loss + self.beta * (fusion_loss / max(present_modalities, 1)) + l_cons * self.gamma
        self.log("train_loss", loss)

        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.student.parameters(), lr=self.lr)


if __name__ == '__main__':
    stud = get_interleaved_EEG_AVI(target_size=384, supporting_latent_size=384)
    teach = MaskedContrastiveModel(hidden_channels=200, out_channels=100)
    teach.load_state_dict(torch.load("../../dependencies/VATE/best_model_contrastive.pt"))
    teach.eval()

    module = EegAviKdVateMaskedModule(stud, teach)

    dataset = FlexibleEmbeddingsSpecMediaDataset("../../data/AMIGOS/p-interleaved-d/spec.csv", cache_in_ram=True)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    vate_dataset = FlexibleEmbeddingsSpecMediaDataset("../../data/AMIGOS/VATE/spec.csv", cache_in_ram=True)

    train_loaders = CombinedLoader({
        # TODO Fare in modo che i due dataloader abbiano stesso seed sempre. Cosi da risultare corrette le draws
        #           o fare wrapper dataset (funziona anche questo).
        "sup": dataloader, "kd": DataLoader(vate_dataset, batch_size=4, shuffle=True)
    }, mode="max_size_cycle")

    trainer = pl.Trainer(accelerator="gpu", devices=1, max_epochs=10)
    trainer.fit(module, train_loaders)

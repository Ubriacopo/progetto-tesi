import lightning as pl
import torch
from lightning.pytorch.utilities import CombinedLoader
from lightning.pytorch.utilities.types import STEP_OUTPUT
from torch import nn
from torch.utils.data import DataLoader

from core_data.dataset import FlexibleEmbeddingsSpecMediaDataset
from model.EEGAVI.EEGAVI import EEGAVI
from model.EEGAVI.interleaved_EEGAVI.interleaved_model import get_interleaved_EEG_AVI
from model.VATE.constrastive_model import ContrastiveModel
import torch.nn.functional as F


class EegAviKdModule(pl.LightningModule):
    def __init__(self, student: EEGAVI, teacher: ContrastiveModel, kd_loss_fn, ce_loss_fn=None,
                 alpha: float = 0.5, lr: float = 1e-4):
        super().__init__()
        self.student = student
        self.teacher = teacher

        for p in self.teacher.parameters():
            p.requires_grad = False

        self.kd_loss_fn = kd_loss_fn
        self.ce_loss_fn = ce_loss_fn
        self.alpha = alpha
        self.lr = lr
        self.tau = 0.2  # Temperature

    def measure_modality_kd_loss(self, teacher_x: torch.Tensor, teacher_mask: torch.Tensor,
                                 student_x: torch.Tensor, student_mask: torch.Tensor) -> torch.Tensor:
        idx = (student_mask.any(dim=1).squeeze() & teacher_mask.bool()).nonzero(as_tuple=True)[0]
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
        x_teacher = batch["kd"]
        x_student = batch["sup"]

        with torch.inference_mode():
            # Get Knowledge Distillation terms
            kd_vid, kd_aud, kd_txt, _ = self.teacher(
                x_teacher["vid"] if "vid" in x_teacher else None,
                x_teacher["aud"] if "aud" in x_teacher else None,
                x_teacher["txt"] if "txt" in x_teacher else None
            )

        stud_out = self.student(x_student, use_kd=True)

        kd_loss = .0
        kd_outs = stud_out["kd_outs"]

        if "vid" in kd_outs:
            kd_loss += self.measure_modality_kd_loss(
                teacher_x=kd_vid, teacher_mask=torch.ones(kd_vid.shape[0], device=kd_vid.device),
                student_x=kd_outs["vid"]["data"], student_mask=kd_outs["vid"]["mask"]
            )

        if "aud" in kd_outs:
            kd_loss += self.measure_modality_kd_loss(
                teacher_x=kd_aud, teacher_mask=torch.ones(kd_aud.shape[0], device=kd_aud.device),
                student_x=kd_outs["aud"]["data"], student_mask=kd_outs["aud"]["mask"]
            )

        if "txt" in kd_outs:
            kd_loss += self.measure_modality_kd_loss(
                teacher_x=kd_txt, teacher_mask=torch.ones(kd_txt.shape[0], device=kd_txt.device),
                student_x=kd_outs["txt"]["data"], student_mask=kd_outs["txt"]["mask"]
            )

        # Apply SigLipLoss now

        loss = self.alpha * kd_loss + self.beta * 0  # TODO

        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.student.parameters(), lr=self.lr)


if __name__ == '__main__':
    stud = get_interleaved_EEG_AVI(target_size=384, supporting_latent_size=384)
    teach = ContrastiveModel(hidden_channels=200, out_channels=100)
    teach.load_state_dict(torch.load("../../dependencies/VATE/best_model_contrastive.pt"))
    teach.eval()

    module = EegAviKdModule(stud, teach, kd_loss_fn=nn.MSELoss(), ce_loss_fn=nn.CrossEntropyLoss())

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

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

    def training_step(self, batch: dict[str, dict], batch_idx) -> STEP_OUTPUT:
        x_teacher = batch["kd"]
        x_student = batch["sup"]

        with torch.inference_mode():
            kd_vid, kd_aud, kd_txt, _ = self.teacher(
                x_teacher["vid"] if "vid" in x_teacher else None,
                x_teacher["aud"] if "aud" in x_teacher else None,
                x_teacher["txt"] if "txt" in x_teacher else None
            )

        stud_out = self.student(x_student, use_kd=True)

        kd_loss = .0
        kd_outs = stud_out["kd_outs"]
        if "vid" in kd_outs:
            # TODO Masking
            # InfoNCE contrastive loss
            x = F.normalize(kd_outs["vid"]["data"], dim=-1)
            t = F.normalize(kd_vid, dim=-1)

            # Pass from (B, P) mask to (B) mask. nvm get a mask already from model
            mask_row = kd_outs["vid"]["mask"]
            mask_col = has_teacher

            # Dot product similarity (Cosine)
            logits = (x @ t.T) / self.tau
            targets = torch.arange(x.size(0), device=x.device)
            kd_loss += F.cross_entropy(logits, targets)

        if "aud" in kd_outs:
            # InfoNCE contrastive loss
            x = F.normalize(kd_outs["aud"]["data"], dim=-1)
            t = F.normalize(kd_aud, dim=-1)
            # Dot product similarity (Cosine)
            logits = (x @ t.T) / self.tau
            targets = torch.arange(x.size(0), device=x.device)
            kd_loss += F.cross_entropy(logits, targets)

        if "txt" in kd_outs:
            # InfoNCE contrastive loss
            x = F.normalize(kd_outs["txt"]["data"], dim=-1)
            t = F.normalize(kd_txt, dim=-1)
            # Dot product similarity (Cosine)
            logits = (x @ t.T) / self.tau
            targets = torch.arange(x.size(0), device=x.device)
            kd_loss += F.cross_entropy(logits, targets)

        # Apply SigLipLoss

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

    # TODO kd dataloader (VATE)

    train_loaders = CombinedLoader({"kd": dataloader, "sup": dataloader}, mode="max_size_cycle")

    trainer = pl.Trainer(accelerator="gpu", devices=1, max_epochs=10)
    trainer.fit(module, train_loaders)

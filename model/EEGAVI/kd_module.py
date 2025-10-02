import lightning as pl
import torch
from lightning.pytorch.utilities import CombinedLoader
from lightning.pytorch.utilities.types import STEP_OUTPUT
from torch import nn
from torch.utils.data import DataLoader

from core_data.dataset import FlexibleEmbeddingsSpecMediaDataset
from model.EEGAVI.interleaved_EEGAVI.interleaved_model import get_interleaved_EEG_AVI
from model.VATE.constrastive_model import ContrastiveModel


class KDModule(pl.LightningModule):
    def __init__(self, student: nn.Module, teacher: ContrastiveModel, kd_loss_fn, ce_loss_fn=None,
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

    def training_step(self, batch, batch_idx) -> STEP_OUTPUT:
        x_teacher = batch["kd"]
        x_student = batch["sup"]

        with torch.inference_mode():
            t_out = self.teacher(
                x_teacher["video"] if "video" in x_teacher else None,
                x_teacher["audio"] if "audio" in x_teacher else None,
                x_teacher["text"] if " text" in x_teacher else None
            )

        s_out = self.student(x_student)
        kd_loss = self.kd_loss_fn(s_out, t_out)

        loss = kd_loss
        if self.ce_loss_fn is not None:
            ce_loss = self.ce_loss_fn(s_out)
            loss = (1 - self.alpha) * kd_loss + self.alpha * ce_loss

        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.student.parameters(), lr=self.lr)


if __name__ == '__main__':
    stud = get_interleaved_EEG_AVI(target_size=384, supporting_latent_size=384)
    teach = ContrastiveModel(hidden_channels=200, out_channels=100)
    teach.load_state_dict(torch.load("../../dependencies/VATE/best_model_contrastive.pt"))
    teach.eval()

    module = KDModule(stud, teach, kd_loss_fn=nn.MSELoss(), ce_loss_fn=nn.CrossEntropyLoss())

    dataset = FlexibleEmbeddingsSpecMediaDataset("../../data/AMIGOS/p-interleaved-d/spec.csv", cache_in_ram=True)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    # TODO kd dataloader (VATE)

    train_loaders = CombinedLoader({"kd": dataloader, "sup": dataloader}, mode="max_size_cycle")

    trainer = pl.Trainer(accelerator="gpu", devices=1, max_epochs=10)
    trainer.fit(module, train_loaders)

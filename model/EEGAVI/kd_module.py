from typing import Literal

import lightning as pl
import torch
import torch.nn.functional as F
from lightning.pytorch.utilities import CombinedLoader
from lightning.pytorch.utilities.types import STEP_OUTPUT
from torch.utils.data import DataLoader

from core_data.dataset import FlexibleEmbeddingsSpecMediaDataset
from model.EEGAVI.EEGAVI import EEGAVI, EEGAVIOutputs
from model.EEGAVI.interleaved_EEGAVI.interleaved_model import get_interleaved_EEG_AVI
from model.VATE.constrastive_model import MaskedContrastiveModel, MaskedContrastiveModelOutputs
from model.loss import siglip, masked_info_nce, masked_cosine_similarity

# TODO non qui ma in generale dovremmo fare un obiettivo semi superivsionato?
# TODO vedi callback utili
class EegAviKdVateMaskedModule(pl.LightningModule):
    def __init__(self, student: EEGAVI, teacher: MaskedContrastiveModel,
                 alpha: float = 1.0, beta: float = 1., gamma: float = 0.5, lr: float = 1e-4):
        super().__init__()
        self.student: EEGAVI = student
        self.teacher: MaskedContrastiveModel = teacher

        # Weight terms for loss parts
        self.alpha: float = alpha
        self.beta: float = beta
        self.gamma: float = gamma

        self.lr = lr
        # Temperature
        self.tau = 0.2

    def configure_optimizers(self):
        # TODO Scheduling lr?
        return torch.optim.AdamW(self.student.parameters(), lr=self.lr)

    def training_step(self, batch: dict[str, dict], batch_idx) -> STEP_OUTPUT:
        stud_out: EEGAVIOutputs = self.student(batch["sup"], use_kd=True)
        with torch.inference_mode():
            teacher_out: MaskedContrastiveModelOutputs = self.teacher(**batch["kd"])

        # KD Loss
        kd_loss = .0
        key: Literal['vid', 'txt', 'aud']
        for key in teacher_out.keys():
            if key not in stud_out.kd_outs:
                continue  # Keys do never match so no kd on this.

            kd_loss += masked_info_nce(
                za=stud_out.kd_outs[key]["data"], za_mask=stud_out.kd_outs[key]["mask"],
                zb=teacher_out[key]["data"], zb_mask=teacher_out[key]["mask"], mask_idx_match=(1, 0), tau=self.tau
            )

        # Fusion Loss
        fusion_loss = .0
        present_modalities: int = 0
        for key, value in stud_out.multimodal_outs.items():
            z, mask = value["data"], value["mask"]
            if mask.dim() == 3:
                mask = mask[:, :, 0].squeeze()
            # Usually stud_out_multimodal out: (b, T, P, D) (Differs for EEG)
            # Patch mean: (Has no mask patches are always max rank).
            if z.dim() > 3:
                # We have patch tokens so we mean over them
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
            l_cons = masked_cosine_similarity(
                za=stud_out.multimodal_outs["eeg"]["data"],
                # ECG goes through the Perceiver Resampler so it is 4D
                zb=stud_out.multimodal_outs["ecg"]["data"].mean(dim=-2),
                present=stud_out.multimodal_outs["eeg"]["mask"] & stud_out.multimodal_outs["ecg"]["mask"].any(dim=-1)
            )

        loss = self.alpha * kd_loss + self.beta * (fusion_loss / max(present_modalities, 1)) + l_cons * self.gamma
        self.log("train_loss", loss)

        return loss


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

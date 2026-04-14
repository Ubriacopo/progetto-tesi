import torch

from main.model.downstream.core.probe_model import SimpleFineTuneProbe
from main.model.downstream.core.trainer.base import AbstractClassificationTrainer
from main.model.neegavi.adapters import EegCbraModAdapter


class EavClassificationTrainer(AbstractClassificationTrainer):
    def __init__(self, model: SimpleFineTuneProbe, seed: int, lr=3e-4, classes: int = 2,
                 backbone_lr=1e-4, cbramod_lr=1e-5, weight_decay: float = 0.01):
        super().__init__(model, seed, lr, classes, backbone_lr)

    def extract_target(self, batch):
        return batch["assessment", "score"][:, 0]

    def configure_optimizers(self):
        m: SimpleFineTuneProbe = self.model
        cbra_adapter: EegCbraModAdapter = m.backbone.pivot.adapter
        cbramod_tail_params = list(cbra_adapter.encoder.proj_out.parameters())
        for layer in cbra_adapter.encoder.encoder.layers[-2:]:
            cbramod_tail_params += list(layer.parameters())

        tail_ids = {id(p) for p in cbramod_tail_params}
        eegavi_params = [p for p in m.backbone.parameters() if p.requires_grad and id(p) not in tail_ids]
        project_params = list(m.project.parameters())

        optimizer = torch.optim.AdamW([
            {"params": project_params, "lr": self.hparams.lr},
            {"params": eegavi_params, "lr": self.hparams.backbone_lr},
            {"params": cbramod_tail_params, "lr": self.hparams.cbramod_lr},
        ], weight_decay=self.hparams.weight_decay, )

        return optimizer

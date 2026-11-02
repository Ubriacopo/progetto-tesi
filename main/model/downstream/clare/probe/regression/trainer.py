from main.model.downstream.core.trainer.regression import RegressionTrainer


class ClaireTrainer(RegressionTrainer):
    def __init__(self, model, seed, out_dim=1, lr=3e-4, backbone_lr=3e-5):
        super().__init__(model, seed, out_dim=out_dim, lr=lr, backbone_lr=backbone_lr)

    def extract_target(self, batch):
        return (batch["assessment", "scores"].float() - 1) / 8

from main.model.downstream.core.trainer.regression import RegressionTrainer


class ClaireTrainer(RegressionTrainer):
    def extract_target(self, batch):
        return (batch["assessment", "scores"][:, 0].float() - 1) / 9

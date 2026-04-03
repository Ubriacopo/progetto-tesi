from main.model.downstream.core.trainer.regression import RegressionTrainer


class FusionTrainer(RegressionTrainer):
    def extract_target(self, batch):
        return (batch["assessment", "scores"][:,3].float() - 1) / 8

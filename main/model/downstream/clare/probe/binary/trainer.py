from main.model.downstream.core.trainer.base import AbstractClassificationTrainer


class ClaireClassificationTrainer(AbstractClassificationTrainer):
    def extract_target(self, batch):
        return (batch["assessment", "scores"].float() >= 5.).long()

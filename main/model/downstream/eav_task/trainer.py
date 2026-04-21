from main.model.downstream.core.trainer.base import AbstractEegAviClassificationTrainer


class EavClassificationTrainer(AbstractEegAviClassificationTrainer):
    def extract_target(self, batch):
        return batch["assessment", "score"][:, 0]

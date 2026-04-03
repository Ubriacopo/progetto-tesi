from main.model.downstream.core.trainer.base import AbstractClassificationTrainer


class FacedTrainer(AbstractClassificationTrainer):
    def extract_target(self, batch):
        return batch["assessment", "score"][:, 0]

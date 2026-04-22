from main.model.downstream.core.trainer.base import AbstractEegAviClassificationTrainer, \
    AbstractCBraClassificationTrainer


class EavClassificationTrainer(AbstractEegAviClassificationTrainer):
    def extract_target(self, batch):
        return batch["assessment", "score"][:, 0]

class EavCbraModClassificationTrainer(AbstractCBraClassificationTrainer):
    def extract_target(self, batch):
        return batch["assessment", "score"][:, 0]

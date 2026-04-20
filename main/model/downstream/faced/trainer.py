from main.model.downstream.core.trainer.base import AbstractClassificationTrainer, AbstractEegAviClassificationTrainer, \
    AbstractCBraClassificationTrainer


class CBraModFacedClassificationTrainer(AbstractCBraClassificationTrainer):
    def extract_target(self, batch):
        return batch["assessment", "score"][:, 0]


class FacedClassificationTrainer(AbstractEegAviClassificationTrainer):
    def extract_target(self, batch):
        return batch["assessment", "score"][:, 0]

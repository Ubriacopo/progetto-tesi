import tensordict
from tensordict import TensorDict

from main.model.downstream.linear_probe_datamodule import LinearProbeDataModule

# TODO Usare h5py
# AMIGOS: train | DEP: test
class AmigosDeapFusionDataModule(LinearProbeDataModule):
    MAX_TIME_STEPS: int = 10

    def setup(self, stage: str) -> None:
        if isinstance(self.train_dataset, list):
            self.train_dataset = self.train_dataset[-1]
            self.valid_dataset = self.valid_dataset[-1]
            self.test_dataset = self.test_dataset[-1]

    @classmethod
    def train_collate_fn(cls, batch: list[TensorDict]):
        T = cls.MAX_TIME_STEPS
        batch = [b.exclude("meta", ("assessment", "scales"), ("assessment", "labels"), )[:T] for b in batch]

        for td in batch:
            # Only take the first 5 (V/A/D/L/F) because the train dataset (AMIGOS) has more
            td["assessment", "scores"] = td["assessment", "scores"][:, :5]

        return tensordict.pad_sequence(batch, 0, return_mask="pad_mask")

    @classmethod
    def val_collate_fn(cls, batch):
        return cls.train_collate_fn(batch)

    @classmethod
    def test_collate_fn(cls, batch):
        T = cls.MAX_TIME_STEPS
        batch = [b.exclude("meta", ("assessment", "scales"), ("assessment", "labels"), )[:T] for b in batch]
        return tensordict.pad_sequence(batch, 0, return_mask="pad_mask")

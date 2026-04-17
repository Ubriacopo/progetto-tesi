import dataclasses


@dataclasses.dataclass
class TrainerConfig:
    epochs: int = 50
    batch_size: int = 32


@dataclasses.dataclass
class ModelConfiguration:
    weights_path: str = "/home/jacopo/PycharmProjects/progetto-tesi/main/model/script/outputs/best-2-attn-1-beta/2026-03-27_22-46-58/checkpoints/epochepoch=39-stepstep=102120.ckpt"
    is_baseline: bool = False # If true Is CBra else it is EEGAVI

@dataclasses.dataclass
class SeedConfig:
    dataset_path: str = "/home/jacopo/dataset/EEGAVI/FACED-PROBE/FINETUNE/interleaved-downstream/"

    seed: int = 42
    labels: int = 9

    model_config: ModelConfiguration = dataclasses.field(default_factory=ModelConfiguration)
    trainer_config: TrainerConfig = dataclasses.field(default_factory=TrainerConfig)



import dataclasses

from main.model.neegavi.adapters import PerceiverResamplerConfig


@dataclasses.dataclass(frozen=True)
class ModalityConfig:
    in_size: int
    out_size: int

    timestep_seconds: int


@dataclasses.dataclass(frozen=True)
class KdModalityConfig(ModalityConfig):
    teacher_out_size: int


@dataclasses.dataclass(frozen=True)
class EegModalityConfig(ModalityConfig):
    channels: int


@dataclasses.dataclass(frozen=True)
class PerceiverModalityConfig(ModalityConfig):
    perceiver_resampler_config: PerceiverResamplerConfig


@dataclasses.dataclass(frozen=True)
class MaskedFeedForwardConfig(ModalityConfig):
    mult: int
    dropout: float


@dataclasses.dataclass(frozen=True)
class KdPerceiverModalityConfig(PerceiverModalityConfig, KdModalityConfig):
    pass


@dataclasses.dataclass(frozen=True)
class KdTemporalEncoderConfig(KdModalityConfig):
    max_length: int


@dataclasses.dataclass(frozen=True)
class VideoModalityConfig(ModalityConfig):
    pass

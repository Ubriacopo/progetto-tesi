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

    @staticmethod
    def default(**kwargs):
        return EegModalityConfig(in_size=None, out_size=384, timestep_seconds=1, channels=19)


@dataclasses.dataclass(frozen=True)
class CBraModEegModalityConfig(EegModalityConfig):
    weights_path: str

    @staticmethod
    def default(weights_path: str, **kwargs):
        return CBraModEegModalityConfig(
            in_size=200, out_size=384, timestep_seconds=1, channels=19, weights_path=weights_path
        )


@dataclasses.dataclass(frozen=True)
class PerceiverModalityConfig(ModalityConfig):
    perceiver_resampler_config: PerceiverResamplerConfig

    @staticmethod
    def vid_default():
        return PerceiverModalityConfig(
            in_size=768, out_size=384, timestep_seconds=4, perceiver_resampler_config=PerceiverResamplerConfig(768, 2)
        )

    @staticmethod
    def aud_default():
        return PerceiverModalityConfig(
            in_size=768, out_size=384, timestep_seconds=4, perceiver_resampler_config=PerceiverResamplerConfig(768, 2)
        )


@dataclasses.dataclass(frozen=True)
class MaskedFeedForwardConfig(ModalityConfig):
    mult: int
    dropout: float = 0.0

    @staticmethod
    def ecg_default():
        return MaskedFeedForwardConfig(in_size=256, out_size=384, timestep_seconds=4, mult=6)


@dataclasses.dataclass(frozen=True)
class KdPerceiverModalityConfig(PerceiverModalityConfig, KdModalityConfig):
    pass


@dataclasses.dataclass(frozen=True)
class KdTemporalEncoderConfig(KdModalityConfig):
    max_length: int


@dataclasses.dataclass(frozen=True)
class VideoModalityConfig(ModalityConfig):
    pass

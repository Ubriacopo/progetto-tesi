import dataclasses
from abc import ABC, abstractmethod
from typing import Optional, Any, Callable

from main.core_data.media.audio import Audio
from main.core_data.media.ecg import ECG
from main.core_data.media.eeg import EEG
from main.core_data.media.text import Text
from main.core_data.media.video import Video
from main.core_data.media.video.playground import out
from main.model.neegavi.adapters import EegAdapter, PerceiverResamplerAdapter, PerceiverResamplerConfig
from main.model.neegavi.blocks import ModalityStream
from main.model.neegavi.kd import KDHead
from main.model.neegavi.model import EegInterAviModel, EegInterAviModelConfiguration
from main.model.neegavi.xattention import GatedXAttentionFactory, GatedXAttentionCustomArgs


# Todo vedi se statefull
class ModalityStreamFactory:
    def __init__(self):
        self.default_perceiver_config = PerceiverResamplerConfig(
            dim=768, depth=2, heads=12, num_latents=64, dim_head=64
        )

    def perceiver_resampler_adapter(self, modality_code: str, input_size: int, output_size: int,
                                    timestep_seconds: int, adapter_config: PerceiverResamplerConfig = None,
                                    use_kd: bool = True, teacher_out_size: int = None, ):
        kd_head: Optional[KDHead] = None

        if use_kd:
            kd_head = KDHead(input_size=input_size, target_size=teacher_out_size)
        config = self.default_perceiver_config if adapter_config is None else adapter_config
        adapter = PerceiverResamplerAdapter(config, project_out_size=output_size)
        return ModalityStream(modality_code, output_size, adapter, timestep_seconds, kd_head)


def supporting(function):
    function._is_supporting = True
    function._is_pivot = False
    return function


def pivot(function):
    function._is_pivot = True
    function._is_supporting = False
    return function


class AbstractEegInterAviFactory(ABC):
    def __init__(self, dim: int, latent_dim: int, custom_config: EegInterAviModelConfiguration = None):
        self.attn_factory = GatedXAttentionFactory(dim, latent_dim)
        self.config = custom_config if custom_config is not None else EegInterAviModelConfiguration()
        self.default_modality_stream_factory = ModalityStreamFactory()

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cls.supporting_methods = cls._collect_marked("_is_supporting")
        pivot_methods = cls._collect_marked("_is_pivot")
        if len(pivot_methods) != 1:
            raise ValueError(f"Expected exactly 1 pivot, found {len(pivot_methods)}.")

        cls.pivot_method = pivot_methods[0]

    @classmethod
    def _collect_marked(cls, attr_name: str):
        out = {}
        # base -> subclass, so subclass overrides win
        for c in reversed(cls.mro()):
            if c is object:
                continue
            for name, obj in c.__dict__.items():
                if callable(obj) and getattr(obj, attr_name, False):
                    out[name] = obj
        return list(out.values())

    def _make_pivot_modality_stream(self, *args, **kwargs):
        return self.pivot_method(self, *args, **kwargs)

    def _make_supporting_modality_stream(self, *args, **kwargs):
        return [i(self, *args, **kwargs) for i in self.supporting_methods]

    def build(self, layers: int, *args, **kwargs):
        return EegInterAviModel(
            self._make_pivot_modality_stream(*args, **kwargs),
            *self._make_supporting_modality_stream(*args, **kwargs),
            # TODO in un secondo momento magari fare layers che siano custom args
            attn_blocks=self.attn_factory.build(layers, None), config=self.config
        )


@dataclasses.dataclass(frozen=True)
class ModalityConfig:
    in_size: int
    out_size: int

    timestep_seconds: int


@dataclasses.dataclass(frozen=True)
class EegModalityConfig(ModalityConfig):
    channels: int


# todo vedi si sistemare
class DefaultEegInterAviFactory(AbstractEegInterAviFactory):
    def __init__(self, dim: int, latent_dim: int,
                 eeg_config: EegModalityConfig,
                 custom_config: EegInterAviModelConfiguration = None):
        super().__init__(dim, latent_dim, custom_config)
        self.modality_stream_factory = ModalityStreamFactory()

        # Dataclass for each?> todo. Potrei fare in modo che sia kwrgs? ma forse cosi é piu semplice?
        #   Pro di qkargs: ho meno parametri salvati
        #   Pro di vasriable: So chiaramete cosa ho leggendo la classe (beh lo ho comunque)
        self.eeg_modality_config = eeg_config

        # todo paraemtrizza
        self.support_in_size: int = 200
        self.support_out_size: int = dim

    @pivot
    def eeg(self, channels: int) -> ModalityStream:
        in_size = self.eeg_modality_config.in_size
        out_size = self.eeg_modality_config.out_size
        return ModalityStream(
            EEG.modality_code(), output_size=out_size,
            timestep_seconds=self.eeg_modality_config.timestep_seconds,
            adapter=EegAdapter(channels, latent_input_size=in_size, output_size=out_size)
        )

    @supporting
    def vid(self, in_size: int, out_size: int, teacher_out_size: int, adapter_config: PerceiverResamplerConfig = None) \
            -> ModalityStream:
        return self.modality_stream_factory.perceiver_resampler_adapter(
            Video.modality_code(), in_size, out_size, teacher_out_size, adapter_config
        )

    @supporting
    def aud(self, in_size: int, out_size: int, teacher_out_size: int, adapter_config: PerceiverResamplerConfig = None) \
            -> ModalityStream:
        return self.modality_stream_factory.perceiver_resampler_adapter(
            Audio.modality_code(), in_size, out_size, teacher_out_size, adapter_config
        )

    @supporting
    def txt(self) -> ModalityStream:
        pass

    @supporting
    def ecg(self) -> ModalityStream:
        pass

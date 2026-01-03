from abc import ABC, abstractmethod

from main.core_data.media.audio import Audio
from main.core_data.media.ecg import ECG
from main.core_data.media.eeg import EEG
from main.core_data.media.text import Text
from main.core_data.media.video import Video
from main.model.neegavi.adapters import EegAdapter, PerceiverResamplerAdapter, TemporalEncoderAdapter, \
    SimpleFeedForwardAdapter
from main.model.neegavi.blocks import ModalityStream
from main.model.neegavi.config import EegModalityConfig, KdPerceiverModalityConfig, PerceiverModalityConfig, \
    MaskedFeedForwardConfig
from main.model.neegavi.kd import KDHead
from main.model.neegavi.model import EegInterAviModel, EegInterAviModelConfiguration, WeaklySupervisedEegInterAviModel, \
    WeaklySupervisedWrapperModelConfiguration
from main.model.neegavi.xattention import GatedXAttentionFactory, GatedXAttentionCustomArgs


def supporting(function):
    function._is_supporting = True
    function._is_pivot = False
    return function


def pivot(function):
    function._is_pivot = True
    function._is_supporting = False
    return function


class AbstractEegInterAviFactory(ABC):
    def __init__(self, disabled_supports: list[str], custom_config: EegInterAviModelConfiguration):
        self.config: EegInterAviModelConfiguration = custom_config
        self.disabled_supports: list[str] = disabled_supports

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cls.supporting_methods = cls._collect_marked("_is_supporting")
        pivot_methods = cls._collect_marked("_is_pivot")
        if len(pivot_methods) != 1:
            raise ValueError(f"Expected exactly 1 pivot, found {len(pivot_methods)}.")

        cls.pivot_name = pivot_methods[0].__name__  # store name, not function

    @classmethod
    def _collect_marked(cls, attr_name: str):
        out = {}
        # Base -> Subclass, so subclass overrides win
        for c in reversed(cls.mro()):
            if c is object:
                continue
            for name, obj in c.__dict__.items():
                if callable(obj) and getattr(obj, attr_name, False):
                    out[name] = obj
        return list(out.values())

    def build(self):
        return EegInterAviModel(
            self.config,
            getattr(self, self.pivot_name)(),
            # Suppress not wanted supports via disabled_supports. They have to match the methodname
            *[i(self) for i in self.supporting_methods if i.__name__ not in self.disabled_supports],
            attn_blocks=self.attention(),
            pooling=self.pooling(),
        )

    def pooling(self):
        return None  # Default to

    @abstractmethod
    def attention(self):
        pass


class DefaultEegInterAviFactory(AbstractEegInterAviFactory):
    def __init__(self,
                 # Configs for each different modality
                 eeg_config: EegModalityConfig,
                 vid_config: KdPerceiverModalityConfig,
                 aud_config: KdPerceiverModalityConfig,
                 txt_config: KdPerceiverModalityConfig,
                 ecg_config: MaskedFeedForwardConfig,
                 disabled_supports: list[str],
                 # Attention config
                 attention_config: int | list[GatedXAttentionCustomArgs],
                 # Model wide configuration
                 custom_config: EegInterAviModelConfiguration = None):
        # If custom config does not exist make it based on known information.
        if custom_config is None:
            custom_config = EegInterAviModelConfiguration(eeg_config.out_size, vid_config.out_size)
        super().__init__(disabled_supports, custom_config)
        self.eeg_modality_config: EegModalityConfig = eeg_config
        self.vid_modality_config: KdPerceiverModalityConfig = vid_config
        self.aud_modality_config: KdPerceiverModalityConfig = aud_config
        self.txt_modality_config: KdPerceiverModalityConfig = txt_config
        self.ecg_modality_config: MaskedFeedForwardConfig = ecg_config
        self.attention_config = attention_config

    def attention(self):
        attention = GatedXAttentionFactory(self.config.pivot_dim, self.config.support_dim)
        return attention.build(self.attention_config)

    @pivot
    def eeg(self) -> ModalityStream:
        config = self.eeg_modality_config  # Specific configuration
        adapter = EegAdapter(config.channels, latent_input_size=config.in_size, output_size=config.out_size)

        return ModalityStream(EEG.modality_code(), output_size=config.out_size,
                              timestep_seconds=config.timestep_seconds, adapter=adapter)

    @supporting
    def vid(self) -> ModalityStream:
        config = self.vid_modality_config  # Specific configuration
        kd_head = KDHead(input_size=config.out_size, target_size=config.teacher_out_size)
        adapter = PerceiverResamplerAdapter(
            config.perceiver_resampler_config, project_out_size=config.out_size, in_size=config.in_size
        )

        return ModalityStream(Video.modality_code(), output_size=config.out_size,
                              timestep_seconds=config.timestep_seconds, adapter=adapter, kd_head=kd_head)

    @supporting
    def aud(self) -> ModalityStream:
        config = self.aud_modality_config  # Specific configuration
        kd_head = KDHead(input_size=config.out_size, target_size=config.teacher_out_size)
        adapter = PerceiverResamplerAdapter(
            config.perceiver_resampler_config, project_out_size=config.out_size, in_size=config.in_size
        )

        return ModalityStream(Audio.modality_code(), output_size=config.out_size,
                              timestep_seconds=config.timestep_seconds, adapter=adapter, kd_head=kd_head)

    @supporting
    def txt(self) -> ModalityStream:
        config = self.txt_modality_config
        kd_head = KDHead(input_size=config.out_size, target_size=config.teacher_out_size)
        # todo parametrizza correttamente
        adapter = TemporalEncoderAdapter(
            dim=config.in_size, max_length=32, timestep_duration=config.timestep_seconds,
            modality=self.config.modality.mode
        )

        return ModalityStream(Text.modality_code(), output_size=config.out_size,
                              timestep_seconds=config.timestep_seconds, adapter=adapter, kd_head=kd_head)

    @supporting
    def ecg(self) -> ModalityStream:
        config = self.ecg_modality_config
        adapter = SimpleFeedForwardAdapter(config.in_size, config.out_size, mult=config.mult, dropout=config.dropout)
        return ModalityStream(ECG.modality_code(), config.out_size, adapter, config.timestep_seconds, )

    def pooling(self):
        return None


class WeaklySupervisedDefaultEegInterAviFactory(DefaultEegInterAviFactory):

    def __init__(self,
                 wrapper_config: WeaklySupervisedWrapperModelConfiguration,
                 eeg_config: EegModalityConfig,
                 vid_config: KdPerceiverModalityConfig,
                 aud_config: KdPerceiverModalityConfig,
                 txt_config: KdPerceiverModalityConfig,
                 ecg_config: PerceiverModalityConfig,
                 disabled_supports: list[str],
                 attention_config: int | list[GatedXAttentionCustomArgs],
                 custom_config: EegInterAviModelConfiguration = None):
        super().__init__(eeg_config, vid_config, aud_config, txt_config, ecg_config, disabled_supports,
                         attention_config, custom_config)
        self.wrapper_config: WeaklySupervisedWrapperModelConfiguration = wrapper_config

    def build(self):
        return WeaklySupervisedEegInterAviModel(
            super().build(), base_model_out_size=self.eeg_modality_config.out_size,
            hidden_size=self.wrapper_config.hidden_size, output_size=self.wrapper_config.output_size,
        )

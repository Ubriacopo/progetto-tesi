from __future__ import annotations

from torch import nn

from main.core_data.media.audio import Audio
from main.core_data.media.ecg import ECG
from main.core_data.media.eeg import EEG
from main.core_data.media.text import Text
from main.core_data.media.video import Video
from main.model.blocks.attention import AbstractAttentionBlock
from main.model.blocks.dropout import ModalityDropout, BernoulliSupportsModalityDropout
from main.model.blocks.kd import KDHead
from main.model.blocks.modality_stream import ModalityStream
from main.model.blocks.time_masked import TimeMaskSwitchableProperties
from main.model.blocks.xattention import GatedXAttentionFactory, GatedXAttentionCustomArgs
from main.model.neegavi.adapters import EegAdapter, PerceiverResamplerAdapter, TemporalEncoderAdapter, \
    SimpleFeedForwardAdapter, PerceiverResamplerConfig
from main.model.neegavi.config import EegModalityConfig, KdPerceiverModalityConfig, MaskedFeedForwardConfig, \
    ModalityConfig, PerceiverModalityConfig
from main.model.neegavi.model import EegInterAviModel, EegInterAviModelConfiguration
from main.model.neegavi.utils import get_model_ckpt
from main.utils.logging import make_logger


class Factory:
    # The real Factory pattern (Clearer than the one I did before)
    def __init__(self):
        self.logger = make_logger(self.__class__.__name__)
        self._config: EegInterAviModelConfiguration | None = None
        self._pivot: ModalityStream | None = None
        self._attention: list[AbstractAttentionBlock] = []
        self._supports: dict[str, ModalityStream] = dict()
        self._disabled_supports: set[str] = set()
        self._pooling: nn.Module | None = None
        self._dropout: ModalityDropout | None = None

        self.built: bool = False

    def config(self, custom_config: EegInterAviModelConfiguration) -> Factory:
        self._config = custom_config
        return self

    def pivot(self, code: str, config: ModalityConfig, adapter: nn.Module, kd: KDHead | None = None) -> Factory:
        if code in self._supports:
            raise ValueError("Supporting modality cannot have same key as pivot")

        self._pivot = ModalityStream(
            code=code,  # Identifying code for pivot
            output_size=config.out_size,  # What sizes to map to
            timestep_seconds=config.timestep_seconds,  # How many seconds a timestep is
            adapter=adapter,
            kd_head=kd  # KD projection head if KD is enabled for the modality
        )

        return self

    def support(self, code: str, config: ModalityConfig, adapter: nn.Module, kd: KDHead | None = None) -> Factory:
        if code in self._supports:
            self.logger.warning(f"You are overriding the config with key: {code}. It has an existing configuration")

        if self._pivot is not None and self._pivot.code == code:
            raise ValueError("Supporting modality cannot have same key as pivot")

        self._supports[code] = ModalityStream(
            code=code,  # Identifying code for pivot
            output_size=config.out_size,  # What sizes to map to
            timestep_seconds=config.timestep_seconds,  # How many seconds a timestep is
            adapter=adapter,
            kd_head=kd  # KD projection head if KD is enabled for the modality
        )

        return self

    def modality_dropout(self, dropout: ModalityDropout) -> Factory:
        self._dropout = dropout
        return self

    def disabled(self, code: str) -> Factory:
        self._disabled_supports.add(code)
        return self

    def attention(self, attention_module: list[AbstractAttentionBlock]) -> Factory:
        self._attention = attention_module
        return self

    def pooling(self, pooling_module: nn.Module) -> Factory:
        self._pooling = pooling_module
        return self

    def _default_attention(self):
        return GatedXAttentionFactory(self._config.pivot_dim, self._config.support_dim).build(2)

    def build(self):
        if self.built:
            raise PermissionError("Factory already built an object")
        if self._config is None:
            raise ValueError("No config initialized. Set it before building.")
        if self._pivot is None:
            raise ValueError("No pivot modality initialized. Set it before building.")
        if len(self._supports.values()) == 0:
            raise ValueError("No support modality initialized. Set it before building.")

        self.built = True
        return EegInterAviModel(
            self._config,
            self._pivot,
            *[value for key, value in self._supports.items() if key not in self._disabled_supports],
            attn_blocks=self._attention if len(self._attention) > 0 else self._default_attention(),
            pooling=self._pooling,  # This can be None
            modality_dropout=self._dropout  # This can be None
        )

    @staticmethod
    def default(
            eeg_config: EegModalityConfig,
            vid_config: KdPerceiverModalityConfig,
            aud_config: KdPerceiverModalityConfig,
            txt_config: KdPerceiverModalityConfig,
            ecg_config: MaskedFeedForwardConfig,
            attention_config: int | list[GatedXAttentionCustomArgs],
            custom_config: EegInterAviModelConfiguration = None,
            disabled_supports: set[str] = None,
    ) -> Factory:
        factory = (
            Factory()
            .config(custom_config)
            # .pooling(None) Pooling is by default None which is a valid value
            .modality_dropout(BernoulliSupportsModalityDropout(4, custom_config.drop_p))  # TODO find good configuration
            .attention(
                GatedXAttentionFactory(custom_config.pivot_dim, custom_config.support_dim).build(attention_config))
            .pivot(
                code=EEG.modality_code(),
                adapter=EegAdapter(eeg_config.channels, eeg_config.in_size, eeg_config.out_size),
                config=eeg_config
            )
            .support(
                code=Video.modality_code(),
                adapter=PerceiverResamplerAdapter(
                    vid_config.perceiver_resampler_config, vid_config.in_size, vid_config.out_size
                ),
                config=vid_config,
                kd=KDHead(input_size=vid_config.out_size, target_size=vid_config.teacher_out_size)
            )
            .support(
                code=Audio.modality_code(),
                adapter=PerceiverResamplerAdapter(
                    aud_config.perceiver_resampler_config, aud_config.in_size, aud_config.out_size
                ),
                config=aud_config,
                kd=KDHead(input_size=aud_config.out_size, target_size=aud_config.teacher_out_size)
            )
            .support(
                code=Text.modality_code(),
                adapter=TemporalEncoderAdapter(
                    txt_config.in_size, 32, txt_config.timestep_seconds, modality=custom_config.modality
                ),
                config=txt_config,
                kd=KDHead(input_size=txt_config.out_size, target_size=txt_config.teacher_out_size)
            )
            .support(
                code=ECG.modality_code(),
                adapter=SimpleFeedForwardAdapter(
                    ecg_config.in_size, ecg_config.out_size, mult=ecg_config.mult, dropout=ecg_config.dropout
                ),
                config=ecg_config
            )
        )

        for disabled_support_code in disabled_supports:
            factory.disabled(disabled_support_code)
        return factory

    @staticmethod
    def best_inference(
            eeg_config: EegModalityConfig = EegModalityConfig(
                in_size=200, out_size=384, channels=19, timestep_seconds=1
            ),
            vid_config: PerceiverModalityConfig = PerceiverModalityConfig(
                in_size=768, out_size=384, timestep_seconds=4,
                perceiver_resampler_config=PerceiverResamplerConfig(768, 2)
            ),
            aud_config: PerceiverModalityConfig = PerceiverModalityConfig(
                in_size=768, out_size=384, timestep_seconds=4,
                perceiver_resampler_config=PerceiverResamplerConfig(768, 2)
            ),
            txt_config: ModalityConfig = ModalityConfig(
                in_size=384, out_size=384, timestep_seconds=4
            ),
            ecg_config: MaskedFeedForwardConfig = MaskedFeedForwardConfig(
                in_size=256, out_size=384, timestep_seconds=4, mult=6
            ),
            attention_config: int | list[GatedXAttentionCustomArgs] = 6,
            custom_config: EegInterAviModelConfiguration = EegInterAviModelConfiguration(
                pivot_dim=384, support_dim=384, output_size=384,
                modality=TimeMaskSwitchableProperties("causal")
            ),
            disabled_supports: set[str] = ("txt",)
    ):
        factory = (
            Factory()
            .config(custom_config)
            .attention(
                GatedXAttentionFactory(custom_config.pivot_dim, custom_config.support_dim).build(attention_config)
            )
            .pivot(
                code=EEG.modality_code(),
                adapter=EegAdapter(eeg_config.channels, eeg_config.in_size, eeg_config.out_size),
                config=eeg_config
            )
            .support(
                code=Video.modality_code(),
                adapter=PerceiverResamplerAdapter(
                    vid_config.perceiver_resampler_config, vid_config.in_size, vid_config.out_size
                ),
                config=vid_config,
            )
            .support(
                code=Audio.modality_code(),
                adapter=PerceiverResamplerAdapter(
                    aud_config.perceiver_resampler_config, aud_config.in_size, aud_config.out_size
                ),
                config=aud_config
            )
            .support(
                code=Text.modality_code(),
                adapter=TemporalEncoderAdapter(
                    txt_config.in_size, 32, txt_config.timestep_seconds, modality=custom_config.modality
                ),
                config=txt_config
            )
            .support(
                code=ECG.modality_code(),
                adapter=SimpleFeedForwardAdapter(
                    ecg_config.in_size, ecg_config.out_size, mult=ecg_config.mult, dropout=ecg_config.dropout
                ),
                config=ecg_config
            )
        )

        for disabled_support_code in disabled_supports:
            factory.disabled(disabled_support_code)

        return factory

    @staticmethod
    def best_inference_loaded(trainer_ckpt_path: str):
        model = Factory.best_inference(disabled_supports=set()).build()
        # Load previous state
        model.load_state_dict(get_model_ckpt(weights_path=trainer_ckpt_path), strict=False)
        # Model is now frozen
        model.eval()
        return model

from main.core_data.media.audio import Audio
from main.core_data.media.ecg import ECG
from main.core_data.media.eeg import EEG
from main.core_data.media.video import Video
from main.model.blocks.dropout import BernoulliSupportsModalityDropout
from main.model.blocks.xattention import GatedXAttentionFactory
from main.model.neegavi.adapters import EegCbraModAdapter, PerceiverResamplerAdapter, SimpleFeedForwardAdapter
from main.model.neegavi.config import EegModalityConfig, PerceiverModalityConfig, MaskedFeedForwardConfig
from main.model.neegavi.factories.core import CoreFactory
from main.model.neegavi.model import EegInterAviModelConfiguration


class FineTuneFactory(CoreFactory):
    @staticmethod
    def fine_tune_default(
            config: EegInterAviModelConfiguration = EegInterAviModelConfiguration(),
            eeg_config: EegModalityConfig = EegModalityConfig.default(),
            vid_config: PerceiverModalityConfig = PerceiverModalityConfig.vid_default(),
            aud_config: PerceiverModalityConfig = PerceiverModalityConfig.aud_default(),
            ecg_config: MaskedFeedForwardConfig = MaskedFeedForwardConfig.ecg_default(),
    ):
        pivot_dim: int = 384
        support_dim: int = 384

        attention_layers: int = 2

        factory = (
            FineTuneFactory()
            # Defaults to best configuration
            .config(config)
            .modality_dropout(BernoulliSupportsModalityDropout(4, 0.1))
            .attention(GatedXAttentionFactory(pivot_dim, support_dim).build(attention_layers))
            .pivot(
                code=EEG.modality_code(), config=eeg_config,
                adapter=EegCbraModAdapter()
            )
            .support(
                code=Video.modality_code(), config=vid_config,
                adapter=PerceiverResamplerAdapter(
                    vid_config.perceiver_resampler_config, vid_config.in_size, vid_config.out_size
                )
            )
            .support(
                code=Audio.modality_code(), config=aud_config,
                adapter=PerceiverResamplerAdapter(
                    aud_config.perceiver_resampler_config, aud_config.in_size, aud_config.out_size
                )
            )
            .support(
                code=ECG.modality_code(), config=ecg_config,
                adapter=SimpleFeedForwardAdapter(
                    ecg_config.in_size, ecg_config.out_size, mult=ecg_config.mult, dropout=ecg_config.dropout
                ),
            )
        )

        return factory

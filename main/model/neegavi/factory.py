from typing import Tuple

from torch import nn

from main.core_data.media.audio import Audio
from main.core_data.media.eeg import EEG
from main.model.EEGAVI.interleaved_EEGAVI.adapters import PerceiverResamplerConfig, EegAdapter, AudioAdapter
from main.model.layer.kd import KDHead
from main.model.neegavi.adapters import AudioAdapter as SimpleAudioAdapter
from main.model.neegavi.base_model import EegInterAviModel, WeaklySupervisedNEEEGBaseModel
from main.model.neegavi.blocks import ModalityStream


class NEEGAviFactory:
    @staticmethod
    def interleaved(target_size: int, supports_latent_size: int, channels: int = 32,
                    teacher_out_shape: Tuple[int, ...] = (1, 100),
                    # Further settings:
                    use_modality_encoder: bool = True, xattn_blocks: int = 2
                    ):
        # TODO Is this problem? Config of PerceiverResampler?
        perceiver_resampler_config = PerceiverResamplerConfig(
            dim=768, depth=2, dim_head=64, heads=6, num_latents=16, max_num_time_steps=34  # dipenda da modality
        )
        return EegInterAviModel(
            output_size=target_size,
            pivot=ModalityStream(
                EEG.modality_code(), target_size,
                adapter=EegAdapter(channels, latent_input_size=200, output_size=target_size),
            ),
            supports=[
                ModalityStream(
                    Audio.modality_code(), target_size,
                    kd_head=KDHead(input_size=supports_latent_size, target_shape=teacher_out_shape,
                                   #transform=nn.Sequential(
                                   #    nn.Linear(supports_latent_size, 128),
                                   #    nn.GELU(),
                                   #    nn.LayerNorm(128),
                                   #    nn.Linear(128, teacher_out_shape[-1]),
                                   #)
                                   ),
                    # adapter=PMAAudioAdapter(project_out_size=target_size),
                    #adapter=AudioAdapter(perceiver_resampler_config, project_out_size=384),
                    adapter=SimpleAudioAdapter(input_size=768, project_out_size=target_size),
                    time_step_length=0.96
                ),
                # ModalityStream(
                #    Text.modality_code(), target_size,
                #    kd_head=KDHead(input_size=supports_latent_size, target_shape=teacher_out_shape),
                #    # adapter=PMAAudioAdapter(project_out_size=target_size),
                #   adapter=TextAdapter(p=64, perceiver_config=perceiver_resampler_config),
                # )
            ],

            use_modality_encoder=use_modality_encoder,
            xattn_blocks=xattn_blocks,
        )

    @staticmethod
    def weak_supervised_interleaved(output_size: int,
                                    base_model_target_size: int, supports_latent_size: int,
                                    channels: int = 32,
                                    teacher_out_shape: Tuple[int, ...] = (1, 100),
                                    # Further settings:
                                    use_modality_encoder: bool = True,
                                    xattn_blocks: int = 2
                                    ):
        return WeaklySupervisedNEEEGBaseModel(
            NEEGAviFactory.interleaved(base_model_target_size, supports_latent_size, channels,
                                       teacher_out_shape, use_modality_encoder, xattn_blocks),
            hidden_size=100,
            output_size=4,
        )

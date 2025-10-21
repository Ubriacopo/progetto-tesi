# todo model factory?
from typing import Tuple

from main.core_data.media.audio import Audio
from main.core_data.media.eeg import EEG
from main.core_data.media.text import Text
from main.model.EEGAVI.base_model import EegBaseModel, WeaklySupervisedEegBaseModel
from main.model.EEGAVI.interleaved_EEGAVI.adapters import EegAdapter, AudioAdapter, PerceiverResamplerConfig, \
    PMAAudioAdapter, TextAdapter
from main.model.layer.kd import KDHead
from main.model.neegavi.blocks import ModalityStream


class EegBaseModelFactory:
    @staticmethod
    def interleaved(target_size: int, supports_latent_size: int, channels: int = 32,
                    teacher_out_shape: Tuple[int, ...] = (1, 100),
                    # Further settings:
                    use_modality_encoder: bool = True,
                    remap_timesteps: int = 32,
                    xattn_blocks: int = 2
                    ):
        # TODO Is this problem? Config of PerceiverResampler?
        perceiver_resampler_config = PerceiverResamplerConfig(
            dim=768, depth=2, dim_head=64, heads=6, num_latents=16, max_num_time_steps=34  # dipenda da modality
        )
        return EegBaseModel(
            output_size=target_size,
            pivot=ModalityStream(
                EEG.modality_code(), target_size,
                adapter=EegAdapter(channels, latent_input_size=200, output_size=target_size),
            ),
            supports=[
                ModalityStream(
                    Audio.modality_code(), target_size,
                    kd_head=KDHead(input_size=supports_latent_size, target_shape=teacher_out_shape),
                 #   adapter=PMAAudioAdapter(project_out_size=target_size),
                    adapter=AudioAdapter(perceiver_config=perceiver_resampler_config, project_out_size=target_size),
                ),
                ModalityStream(
                    Text.modality_code(), target_size,
                    kd_head=KDHead(input_size=supports_latent_size, target_shape=teacher_out_shape),
                    # adapter=PMAAudioAdapter(project_out_size=target_size),
                    adapter=TextAdapter(p=64, perceiver_config=perceiver_resampler_config),
                )
            ],

            use_modality_encoder=use_modality_encoder,
            xattn_blocks=xattn_blocks,
            remap_timesteps=remap_timesteps
        )

    @staticmethod
    def weak_supervised_interleaved(output_size: int,
                                    base_model_target_size: int, supports_latent_size: int,
                                    channels: int = 32,
                                    teacher_out_shape: Tuple[int, ...] = (1, 100),
                                    # Further settings:
                                    use_modality_encoder: bool = True,
                                    remap_timesteps: int = 32,
                                    xattn_blocks: int = 2
                                    ):
        return WeaklySupervisedEegBaseModel(
            EegBaseModelFactory.interleaved(base_model_target_size, supports_latent_size, channels,
                                            teacher_out_shape, use_modality_encoder, remap_timesteps, xattn_blocks),
            hidden_size=output_size * 2,
            output_size=output_size,
        )

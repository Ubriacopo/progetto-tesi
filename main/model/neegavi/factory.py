from abc import ABC, abstractmethod
from typing import Tuple

from main.core_data.media.audio import Audio
from main.core_data.media.eeg import EEG
from main.core_data.media.text import Text
from main.core_data.media.video import Video
from main.model.neegavi.adapters import PerceiverResamplerAdapter, \
    TemporalEncoderAdapter, PerceiverResamplerConfig, EegAdapter
from main.model.neegavi.base_model import EegInterAviModel, WeaklySupervisedNEEEGBaseModel
from main.model.neegavi.blocks import ModalityStream
from main.model.neegavi.kd import KDHead

class EegInterAviFactory:
    @staticmethod
    def build(output_size: int, pivot: ModalityStream, supports: list[ModalityStream],
              use_modality_encoder: bool, xattn_blocks: int, drop_p: float):
        return EegInterAviModel(output_size, pivot, supports, xattn_blocks, drop_p, use_modality_encoder)

    @staticmethod
    def default(target_size: int, supports_latent_size: int, channels: int = 34,
                teacher_out_shape: Tuple[int, ...] = (1, 100),
                # Further settings:
                use_modality_encoder: bool = True, xattn_blocks: int = 2):
        perceiver_resampler_config = PerceiverResamplerConfig(
            dim=768, depth=2, dim_head=64, heads=12, num_latents=64,
            max_num_time_steps=34  # Dipenda da modality (Non piu visto che passiamo a tutti sesso fs)
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
                    kd_head=KDHead(input_size=supports_latent_size, target_shape=teacher_out_shape),
                    adapter=PerceiverResamplerAdapter(perceiver_resampler_config, project_out_size=384)
                ),
                ModalityStream(
                    Video.modality_code(), target_size,
                    kd_head=KDHead(input_size=supports_latent_size, target_shape=teacher_out_shape),
                    adapter=PerceiverResamplerAdapter(perceiver_resampler_config, project_out_size=384)
                ),
                ModalityStream(
                    Text.modality_code(), target_size,
                    kd_head=KDHead(input_size=supports_latent_size, target_shape=teacher_out_shape),
                    adapter=TemporalEncoderAdapter(p=64, dim=384, ),
                )
            ],

            use_modality_encoder=use_modality_encoder,
            xattn_blocks=xattn_blocks,
        )

    @staticmethod
    def eeg_pooled(output_size: int, supports_latent_size: int,
                   perceiver_resampler_config: PerceiverResamplerConfig,
                   eeg_channels: int = 32,
                   teacher_out_shape: Tuple[int, ...] = (1, 100),
                   # Further settings:
                   use_modality_encoder: bool = True, xattn_blocks: int = 2):
        pivot = ModalityStream(EEG.modality_code(), output_size, adapter=None)  # TODO
        supports = [
            # Audio
            ModalityStream(
                Audio.modality_code(), output_size,
                kd_head=KDHead(input_size=supports_latent_size, target_shape=teacher_out_shape),
                adapter=PerceiverResamplerAdapter(perceiver_resampler_config, project_out_size=384)
            ),
            # Video
            ModalityStream(
                Video.modality_code(), output_size,
                kd_head=KDHead(input_size=supports_latent_size, target_shape=teacher_out_shape),
                adapter=PerceiverResamplerAdapter(perceiver_resampler_config, project_out_size=384)
            ),
            # Text
            ModalityStream(
                Text.modality_code(), output_size,
                kd_head=KDHead(input_size=supports_latent_size, target_shape=teacher_out_shape),
                # adapter=PMAAudioAdapter(project_out_size=target_size),
                adapter=TemporalEncoderAdapter(p=64, dim=384, ),
            )
            # ECG todo
        ]

        return EegInterAviFactory.build(
            output_size, pivot, supports, use_modality_encoder, xattn_blocks
        )

    @staticmethod
    def debug(target_size: int, supports_latent_size: int, channels: int = 32,
              teacher_out_shape: Tuple[int, ...] = (1, 100),
              # Further settings:
              use_modality_encoder: bool = True, xattn_blocks: int = 2
              ):
        # TODO Is this problem? Config of PerceiverResampler? TODO mi manca memoria?
        perceiver_resampler_config = PerceiverResamplerConfig(
            dim=768, depth=2, dim_head=64, heads=12, num_latents=64, max_num_time_steps=34  # dipenda da modality
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
                                   # transform=nn.Sequential(
                                   #    nn.Linear(supports_latent_size, 128),
                                   #    nn.GELU(),
                                   #    nn.LayerNorm(128),
                                   #    nn.Linear(128, teacher_out_shape[-1]),
                                   # )
                                   ),
                    # adapter=PMAAudioAdapter(project_out_size=target_size),
                    adapter=PerceiverResamplerAdapter(perceiver_resampler_config, project_out_size=384),
                    # adapter=SimpleAudioAdapter(input_size=768, project_out_size=target_size),
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
                                    channels: int = 34,
                                    teacher_out_shape: Tuple[int, ...] = (1, 100),
                                    # Further settings:
                                    use_modality_encoder: bool = True,
                                    xattn_blocks: int = 2
                                    ):
        return WeaklySupervisedNEEEGBaseModel(
            EegInterAviFactory.default(base_model_target_size, supports_latent_size, channels,
                                       teacher_out_shape, use_modality_encoder, xattn_blocks),
            hidden_size=100, output_size=4,
        )

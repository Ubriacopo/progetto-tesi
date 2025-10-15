# todo model factory?
from typing import Tuple

from main.core_data.media.audio import Audio
from main.core_data.media.eeg import EEG
from main.model.EEGAVI.base_model import EegBaseModel
from main.model.EEGAVI.interleaved_EEGAVI.adapters import EegAdapter, AudioAdapter, PerceiverResamplerConfig
from main.model.layer.kd import KDHead
from main.model.layer.modality_stream import ModalityStream


class EeeBaseModelFactory:
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
            dim=768, depth=2, dim_head=64, heads=6, num_latents=16
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
                    adapter=AudioAdapter(perceiver_resampler_config, project_out_size=target_size),
                )
            ],

            use_modality_encoder=use_modality_encoder,
            xattn_blocks=xattn_blocks,
            remap_timesteps=remap_timesteps
        )

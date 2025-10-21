from torch.utils.data import DataLoader

from main.core_data.dataset import FlexibleEmbeddingsSpecMediaDataset
from main.core_data.media.audio import Audio
from main.core_data.media.ecg import ECG
from main.core_data.media.eeg import EEG
from main.core_data.media.text import Text
from main.core_data.media.video import Video
from main.model.EEGAVI.EEGAVI import EEGAVI, WeaklySupervisedEEGAVI
from main.model.EEGAVI.interleaved_EEGAVI.adapters import VideoAdapter, PerceiverResamplerConfig, AudioAdapter, \
    TextAdapter, \
    EegAdapter
from main.model.layer.kd import KDHead
from main.model.neegavi.blocks import ModalityStream


# todo factory?
# todo also fix args
def get_interleaved_EEG_AVI(target_size: int, supporting_latent_size: int):
    vate_out_shape = (1, 100)
    channels = 32
    return EEGAVI(
        pivot_latent_size=target_size,
        pivot_modality=ModalityStream(
            code=EEG.modality_code(),
            adapter=EegAdapter(channels=channels, latent_input_size=200, output_size=target_size),
        ),
        supporting_latent_size=supporting_latent_size,
        supporting_modalities=[
            #ModalityStream(
            #    code=Video.modality_code(),
            #    kd_head=KDHead(input_size=supporting_latent_size, target_shape=vate_out_shape),
            #    adapter=VideoAdapter(
            #        PerceiverResamplerConfig(dim=768, depth=2, dim_head=64, heads=6, num_latents=16),
            #        project_out_size=384, patch_size=1
            #    )
            #),
            ModalityStream(
                code=Audio.modality_code(),
                kd_head=KDHead(input_size=supporting_latent_size, target_shape=vate_out_shape),
                adapter=AudioAdapter(
                    PerceiverResamplerConfig(dim=768, depth=2, dim_head=64, heads=6, num_latents=16),
                    project_out_size=384
                )
            ),
            #ModalityStream(
            #    code=Text.modality_code(),
            #    kd_head=KDHead(input_size=supporting_latent_size, target_shape=vate_out_shape),
            #   adapter=TextAdapter(
            #        64,
            #        PerceiverResamplerConfig(dim=384, depth=4, dim_head=64, heads=6, num_latents=16),
            #        project_out_size=384
            #    )
            #),
            #ModalityStream(
            #    code=ECG.modality_code(),
            #    adapter=VideoAdapter(
            #        PerceiverResamplerConfig(dim=256, depth=2, dim_head=64, heads=6, num_latents=16),
            #        project_out_size=384, patch_size=1
            #    )
            #)
        ],

        use_modality_encoder=True,
        xattn_blocks=4,
        remap_timesteps=32,
    )


def get_interleaved_weakly_supervised(target_size: int, supporting_latent_size: int,
                                      teacher_out_shape: tuple = (1, 100), channels: int = 32):
    return WeaklySupervisedEEGAVI(
        eeg_avi=get_interleaved_EEG_AVI(target_size, supporting_latent_size),
        hidden_size=target_size * 2, supervised_target_size=4,
    )

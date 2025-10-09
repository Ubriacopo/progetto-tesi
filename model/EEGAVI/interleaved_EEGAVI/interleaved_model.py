from einops.layers.torch import Rearrange
from torch import nn
from torch.utils.data import DataLoader

from core_data.dataset import FlexibleEmbeddingsSpecMediaDataset
from core_data.media.audio import Audio
from core_data.media.ecg import ECG
from core_data.media.eeg import EEG
from core_data.media.text import Text
from core_data.media.video import Video
from model.EEGAVI.EEGAVI import EEGAVI
from model.EEGAVI.interleaved_EEGAVI.adapters import VideoAdapter, PerceiverResamplerConfig, AudioAdapter, TextAdapter, \
    EegAdapter
from model.layer.kd import KDHead
from model.layer.modality_stream import ModalityStream


def get_interleaved_EEG_AVI(target_size: int, supporting_latent_size: int):
    vate_out_shape = (1, 100)
    channels = 14
    return EEGAVI(
        pivot_latent_size=target_size,
        pivot_modality=ModalityStream(
            code=EEG.modality_code(),
            adapter=nn.Sequential(
                # TODO: Questo smette di funzionare nel momento in cui ho un numero diverso di channels.
                #       Dovrei paddare tutti e avere maschera a channel?
                Rearrange("b T c L -> b T (c L)"),
                nn.LayerNorm(channels * 200),
                nn.Linear(channels * 200, target_size),
                nn.GELU()
            )
        ),
        supporting_latent_size=supporting_latent_size,
        supporting_modalities=[
            ModalityStream(
                code=Video.modality_code(),
                kd_head=KDHead(input_size=supporting_latent_size, target_shape=vate_out_shape),
                # TODO ora che ho fatto downsampling Video Adapter con patch size=16 sarebbe sbagliato.
                #       per dividere dovrei fare? p=1 cosi F =64 corretto?
                adapter=VideoAdapter(
                    PerceiverResamplerConfig(dim=768, depth=2, dim_head=64, heads=6, num_latents=16),
                    project_out_size=384, patch_size=1
                )
            ),
            ModalityStream(
                code=Audio.modality_code(),
                kd_head=KDHead(input_size=supporting_latent_size, target_shape=vate_out_shape),
                adapter=AudioAdapter(
                    PerceiverResamplerConfig(dim=768, depth=2, dim_head=64, heads=6, num_latents=16),
                    project_out_size=384
                )
            ),
            ModalityStream(
                code=Text.modality_code(),
                kd_head=KDHead(input_size=supporting_latent_size, target_shape=vate_out_shape),
                adapter=TextAdapter(
                    64,
                    PerceiverResamplerConfig(dim=384, depth=4, dim_head=64, heads=6, num_latents=16),
                    project_out_size=384
                )
            ),
            ModalityStream(
                code=ECG.modality_code(),
                adapter=VideoAdapter(
                    PerceiverResamplerConfig(dim=256, depth=2, dim_head=64, heads=6, num_latents=16),
                    project_out_size=384, patch_size=1
                )
            )
        ],

        use_modality_encoder=True,
        xattn_blocks=4,
        remap_timesteps=32,
    )


if __name__ == '__main__':
    model = get_interleaved_EEG_AVI(384, 384)
    model.to("cuda:0")
    dataset = FlexibleEmbeddingsSpecMediaDataset("../../../data/AMIGOS/p-interleaved-d/spec.csv", cache_in_ram=True)
    # Problema non riesco a salire sopra batch_size =4
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    # res = model(dataset[0])
    o = next(iter(dataloader))
    res_b = model(o, use_kd=True)
    print(res_b)

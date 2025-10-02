from typing import Optional

from einops.array_api import rearrange
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
from model.layer.perceiver_adapter import PerceiverResampler
from model.layer.utils import DictExtract

# TODO: The dimensionality jump from your frozen encoders (likely 768/1024) to 384 in the adapters might be lossy
# TODO: Sequence length: Your EEG has 85 tokens - is this sufficient temporal resolution? (ViViT has 3306)
# TODO: Perceiver resampler might be overkill. I can just do Linear Projection + LN -> Provare modelli diversi?
#       Questo discorso vale nel momento in cui non ho mai fusioni. Se invece ho ad esmpio clip di 3 secondi con più di 32 frame:
"""
    # Your pipeline becomes:
    video_chunks = chunk_video(variable_video, chunk_size=32)
    video_features = torch.cat([vivit_encoder(chunk) for chunk in video_chunks], dim=1)
    video_adapted = perceiver_adapter(video_features)  # (1, 64, 384) - fixed!

    Chiaramente qui si ha un senso per il perceiver resampler.
"""


def get_interleaved_EEG_AVI(target_size: int, supporting_latent_size: int):
    vate_out_shape = (1, 100)
    c = 14
    return EEGAVI(
        pivot_latent_size=target_size,
        pivot_modality=ModalityStream(
            code=EEG.modality_code(),
            adapter=EegAdapter(c, 200, 384)
        ),
        supporting_latent_size=supporting_latent_size,
        supporting_modalities=[
            ModalityStream(
                code=Video.modality_code(),
                kd_head=KDHead(input_size=supporting_latent_size, target_shape=vate_out_shape),
                adapter=VideoAdapter(PerceiverResamplerConfig(dim=768, depth=2), project_out_size=384)
            ),
            ModalityStream(
                code=Audio.modality_code(),
                kd_head=KDHead(input_size=supporting_latent_size, target_shape=vate_out_shape),
                adapter=AudioAdapter(PerceiverResamplerConfig(dim=768, depth=2), project_out_size=384)
            ),
            ModalityStream(
                code=Text.modality_code(),
                kd_head=KDHead(input_size=supporting_latent_size, target_shape=vate_out_shape),
                adapter=TextAdapter(64, PerceiverResamplerConfig(dim=384, depth=4), project_out_size=384)
            ),
            ModalityStream(
                code=ECG.modality_code(),
                adapter=VideoAdapter(PerceiverResamplerConfig(dim=256, depth=2), project_out_size=384, patch_size=1)
            )
        ],

        use_modality_encoder=True,
        xattn_blocks=4,
        final_projector=nn.Sequential(

        ),
        remap_timesteps=32
    )


if __name__ == '__main__':
    model = get_interleaved_EEG_AVI(384, 384)
    model.to("cuda:0")
    dataset = FlexibleEmbeddingsSpecMediaDataset("../../../data/AMIGOS/p-interleaved-d/spec.csv", cache_in_ram=True)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

    # res = model(dataset[0])
    o = next(iter(dataloader))
    res_b = model(o, use_kd=True)
    print(res_b)

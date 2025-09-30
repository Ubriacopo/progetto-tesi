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
from model.layer.modality_stream import ModalityStream
from model.layer.perceiver_adapter import PerceiverResampler, DictPerceiverResampler
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


class MayArgsOrKwargs(nn.Module):
    def __init__(self, module: nn.Module):
        super().__init__()
        self.module = module  # fn is a Module or callable

    def forward(self, x):
        if isinstance(x, tuple):
            return self.module(*x)  # Unpack tuple
        if isinstance(x, dict):
            return self.module(**x)  # Unpack dictionary
        return self.module(x)


class VideoAdapter(nn.Module):
    def __init__(self, dim: int, depth: int, dim_head: int = 64, heads: int = 8, num_latents: int = 64,
                 max_num_media: int = None, max_num_frames: int = None, ff_mult: int = 4):
        super().__init__()
        self.patch_size = 16
        self.module = PerceiverResampler(dim=dim, depth=depth, dim_head=dim_head, heads=heads, num_latens=num_latents,
                                         max_num_media=max_num_media, max_num_frames=max_num_frames, ff_mult=ff_mult)



    def forward(self, x: dict):
        x, mask = x["data"], None if not "mask" in x else x["mask"]
        x = Rearrange("b T (F p) D -> b T F p D", F=self.patch_size)(x)

        y = self.module(x=x, mask=mask)
        return y


def get_interleaved_EEG_AVI(target_size: int, supporting_latent_size: int):
    vate_out_shape = (1, 100)

    return EEGAVI(
        pivot_latent_size=target_size,
        pivot_modality=ModalityStream(
            code=EEG.modality_code(),
            adapter_output_size=target_size,
            adapter=nn.Sequential(
                DictExtract("data"),
                Rearrange("b c P D -> b D c P"),
                nn.AdaptiveMaxPool2d((1, 1)),  # TODO: masked version of AdaptiveAveragePool2d
                nn.Flatten(start_dim=1),
                Rearrange("(b T) D -> b T D", T=1),
                nn.Linear(200, target_size),
            )
        ),
        supporting_latent_size=supporting_latent_size,
        supporting_modalities=[
            ModalityStream(
                code=Video.modality_code(),
                adapter_output_size=supporting_latent_size,
                kd_shape=vate_out_shape,
                adapter=VideoAdapter(dim=768, depth=4)
            ),
            ModalityStream(
                code=Audio.modality_code(),
                adapter_output_size=supporting_latent_size,
                kd_shape=vate_out_shape,
                adapter=DictPerceiverResampler(dim=768, depth=4)
            ),
            ModalityStream(
                code=Text.modality_code(),
                adapter_output_size=supporting_latent_size,
                kd_shape=vate_out_shape,
                adapter=nn.Sequential(
                    PerceiverResampler(dim=768, depth=4)
                )
            ),
            ModalityStream(
                code=ECG.modality_code(),
                adapter_output_size=supporting_latent_size,
                adapter=DictPerceiverResampler(dim=768, depth=4)
            )
        ],

        use_modality_encoder=True,
        cross_attention_blocks=4,
        final_projector=nn.Sequential(

        )
    )


if __name__ == '__main__':
    model = get_interleaved_EEG_AVI(384, 384)
    model.to("cuda:0")
    dataset = FlexibleEmbeddingsSpecMediaDataset("../../../data/AMIGOS/p-interleaved-d/spec.csv", cache_in_ram=True)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

    # res = model(dataset[0])

    res_b = model(next(iter(dataloader)))

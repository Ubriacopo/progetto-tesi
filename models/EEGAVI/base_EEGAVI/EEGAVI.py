from einops.layers.torch import Rearrange
from torch import nn

from common.model.layers.ISAB import ISAB, PMA
from common.model.layers.modality_stream import ModalityStream
from models.EEGAVI.EEGAVI import EEGAVI


def get_default_simple_EEGAVI():
    target_size = 384
    supporting_size_embedding: int = 768

    vate_out_shape = (1, 100)

    return EEGAVI(
        target_size=target_size,
        pivot_modality=ModalityStream(
            code="eeg",
            adapter=nn.Sequential(
                Rearrange("b c P D -> b D c P"),
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(start_dim=1),
                Rearrange("(b T) D -> b T D", T=1),
                nn.Linear(200, target_size),
            ),
            adapter_output_size=target_size,
        ),
        supporting_size_embedding=supporting_size_embedding,
        supporting_modalities=[
            ModalityStream(
                code="vid",
                adapter=nn.Sequential(
                    ISAB(768, 8, 10),
                    PMA(768, 8, 10),
                ),
                adapter_output_size=768,
                kd_shape=vate_out_shape
            ),
            ModalityStream(
                code="aud",
                adapter=nn.Sequential(
                    PMA(768, 8, 10),
                    nn.Linear(768, supporting_size_embedding),
                ),
                adapter_output_size=supporting_size_embedding
            )
        ],
        use_modality_encoder=True,
        cross_attention_blocks=4,
        final_projector=nn.Sequential(
            nn.LayerNorm(target_size),
            nn.Linear(target_size, target_size * 2),
            nn.GELU(),
            nn.Linear(target_size * 2, target_size)
        )
    )

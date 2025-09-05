from einops.layers.torch import Rearrange
from torch import nn

from common.model.embedding.embedder_adapter import EmbedderAdapter
from common.model.embedding.predefined.cbramod import CBraModFoundationEmbedder
from common.model.embedding.predefined.vivit import ViViTFoundationEmbedder
from common.model.embedding.predefined.w2vbert import W2VBertFoundationEmbedder
from common.model.layers.ISAB import ISAB, PMA
from models.EEGAVI.EEGAVI import EEGAVI


def get_default_simple_EEGAVI():
    video_embedder = ViViTFoundationEmbedder()
    audio_embedder = W2VBertFoundationEmbedder()

    target_size = 384
    supporting_size_embedding: int = 768

    vate_out_shape = (1, 100)

    return EEGAVI(
        target_size=target_size,
        pivot_modality=(
            "eeg",
            EmbedderAdapter(
                embedder=CBraModFoundationEmbedder(),
                adapter=nn.Sequential(
                    Rearrange("b c P D -> b D c P"),
                    nn.AdaptiveAvgPool2d((1, 1)),
                    nn.Flatten(start_dim=1),
                    Rearrange("(b T) D -> b T D", T=1),
                    nn.Linear(200, target_size),
                ),
                kd_shape=vate_out_shape,  # todo remvoe eeg does not go with vate rn
                output_size=target_size
            )
        ),
        supporting_size_embedding=supporting_size_embedding,
        supporting_modalities=[
            (
                "vid",
                EmbedderAdapter(
                    embedder=video_embedder,
                    kd_shape=vate_out_shape,
                    adapter=nn.Sequential(
                        ISAB(video_embedder.output_size, 8, 10),
                        PMA(video_embedder.output_size, 8, 10),
                        # No linear as supporting_size_embedding == video_embedder.output_size
                    ),
                    output_size=supporting_size_embedding
                )
            ),
            (
                "aud",
                # todo new version of embedder adapter? io devo fondere last 3 levels
                EmbedderAdapter(
                    embedder=audio_embedder,
                    kd_shape=vate_out_shape,
                    adapter=nn.Sequential(
                        PMA(audio_embedder.output_size, 8, 10),
                        nn.Linear(audio_embedder.output_size, supporting_size_embedding),
                    ),
                    output_size=supporting_size_embedding
                )
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

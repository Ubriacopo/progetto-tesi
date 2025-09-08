from einops.layers.torch import Rearrange
from torch import nn

from common.data.transform import IDENTITY
from common.model.embedding.embedder_adapter import EmbedderAdapter
from common.model.layers.ISAB import PMA, ISAB
from models.EEGAVI.EEGAVI import EEGAVI


# This model supposes embeddings to be fed to it directly. It does only assume on the shapes.
# As it has to because obvious reasons.
def get_default_core_EEGAVI(
        vid_emb_output_size: int = 768,
        aud_emb_output_size: int = 1024,
):
    target_size = 384
    supporting_size_embedding: int = 768

    vate_out_shape = (1, 100)

    return EEGAVI(
        target_size=target_size,
        pivot_modality=(
            "eeg",
            EmbedderAdapter(
                embedder=None,
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
                    embedder=None,
                    kd_shape=vate_out_shape,
                    adapter=nn.Sequential(
                        ISAB(vid_emb_output_size, 8, 10),
                        PMA(vid_emb_output_size, 8, 10),
                        nn.Linear(vid_emb_output_size, target_size) if vid_emb_output_size != target_size else IDENTITY,
                        # No linear as supporting_size_embedding == video_embedder.output_size
                    ),
                    output_size=supporting_size_embedding
                )
            ),
            (
                "aud",
                EmbedderAdapter(
                    embedder=None,
                    kd_shape=vate_out_shape,
                    adapter=nn.Sequential(
                        PMA(aud_emb_output_size, 8, 10),
                        nn.Linear(aud_emb_output_size, supporting_size_embedding),
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

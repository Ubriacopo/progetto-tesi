import torch
from einops import rearrange
from einops.layers.torch import Rearrange
from torch import nn

from common.model.embedding.embedder_adapter import EmbedderAdapter
from common.model.embedding.predefined.cbramod import CBraModFoundationEmbedder
from common.model.embedding.predefined.w2vbert import W2VBertFoundationEmbedder
from common.model.embedding.predefined.vivit import ViViTFoundationEmbedder
from common.model.embedding.foundation_embedder import FoundationEmbedder
from common.model.layers.attention.x_attention import GatedCrossAttentionBlock
from common.model.layers.ISAB import ISAB, PMA
from models.EEGAVI.EEGAVI import EEGAVI
from models.EEGAVI.transforms import media_locs_single_item


def get_default_simple_EEGAVI():
    video_embedder = ViViTFoundationEmbedder()
    audio_embedder = W2VBertFoundationEmbedder()
    target_size = 384
    supporting_size_embedding: int = 768
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
                ),
                target_size=target_size
            )
        ),
        supporting_size_embedding=supporting_size_embedding,
        supporting_modalities=[
            (
                "vid",
                EmbedderAdapter(
                    embedder=video_embedder,
                    target_size=768,
                    kd_size=100,
                    adapter=nn.Sequential(
                        ISAB(video_embedder.output_size, 8, 10),
                        PMA(video_embedder.output_size, 8, 10),
                    )

                )
            ),
            (
                "aud",
                # todo new version of embedder adapter? io devo fondere last 3 levels
                EmbedderAdapter(
                    embedder=audio_embedder,
                    target_size=supporting_size_embedding,
                    kd_size=100,
                    adapter=nn.Sequential(
                        PMA(audio_embedder.output_size, 8, 10),
                    )
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


class WorkingEEGAVI(nn.Module):
    def __init__(self,
                 target_shape: int = 384,
                 cross_attention_blocks: int = 2,

                 base_video: FoundationEmbedder = ViViTFoundationEmbedder(),
                 video_kd_size: int | None = None,

                 base_audio: FoundationEmbedder = W2VBertFoundationEmbedder(),
                 audio_kd_size: int | None = None,

                 base_eeg: FoundationEmbedder = CBraModFoundationEmbedder(),
                 use_kd: bool = True
                 ):
        super().__init__()
        # If Knowledge distillation is to be used with this module.
        self.use_kd = use_kd

        self.video_adapter = EmbedderAdapter(
            embedder=base_video, target_size=target_shape, kd_size=video_kd_size,
            adapter=nn.Sequential(
                ISAB(base_video.output_size, 8, 10),
                PMA(base_video.output_size, 8, 10)
            )
        )

        self.audio_adapter = EmbedderAdapter(
            embedder=base_audio, target_size=target_shape, kd_size=audio_kd_size,
            adapter=nn.Sequential(
                PMA(base_audio.output_size, 8, 10)
            )
        )

        self.eeg_adapter = EmbedderAdapter(
            embedder=base_eeg, target_size=target_shape,
            # Nothing happens for EEG data
            adapter=nn.Sequential()
        )

        self.gatedXAttn_layers = nn.ModuleList([
            GatedCrossAttentionBlock(dim=target_shape, dim_latent=target_shape)
            for _ in range(cross_attention_blocks)
        ])

        # TODO Revise this and choose a good architecture
        self.projector = nn.Sequential(
            nn.LayerNorm(target_shape),
            nn.Linear(target_shape, target_shape * 2),
            nn.GELU(),
            nn.Linear(target_shape * 2, target_shape)
        )

    # TODO poi forse: Dovrebbe non essere difficile renderlo dinamico con le modalità
    def forward(self, x: dict):
        # Order is unique
        order = list(zip(*x["order"]))[0]
        # By default, don't use KD
        use_kd = "kd" in x and x["kd"]

        mask = x["mask"] if "mask" in x else None
        # Access the stored data:
        eeg = x["eeg"]
        eeg_mask = None

        z_mod = []
        vid = x["vid"] if "vid" in x else None
        vid_mask = mask[:, order.index("vid")] if mask is not None else None

        aud = x["aud"] if "aud" in x else None
        aud_mask = mask[:, order.index("aud")] if mask is not None else None

        z_kd_vid = None
        z_vid = self.video_adapter(vid, vid_mask)
        if isinstance(z_vid, tuple):
            z_vid, z_kd_vid = z_vid
        z_mod.append(z_vid)

        z_kd_aud = None
        z_aud = self.audio_adapter(aud, aud_mask)
        if isinstance(z_aud, tuple):
            z_aud, z_kd_aud = z_aud
        z_mod.append(z_aud)

        z_eeg = self.eeg_adapter(eeg, use_kd)
        b, c, T, D = z_eeg.shape
        # TODO: Vediamo questo poi. Stando a quanto ho visto mi sa che devo unire c e D ?
        z_eeg = rearrange(z_eeg, "b c T D -> b (T c) D")
        media_locations = media_locs_single_item(b, T, z_eeg.device)

        # TODO new mask?
        # TODO Add modality embedding?
        embeddings = torch.cat(z_mod, dim=1)
        if len(embeddings.shape) == 3:
            # (b, T*F, D) (Case of no time series used).
            embeddings = rearrange(embeddings, "b (T F) d -> b T F d", T=1)

        for gated_x_attn in self.gatedXAttn_layers:
            z_eeg = gated_x_attn(z_eeg, embeddings, media_locations=media_locations)

        # TODO: Self attention?

        logits = self.projector(z_eeg)
        return (logits, {"kd_ve": z_kd_vid, "kd_ae": z_kd_aud}) if use_kd else logits

import torch
from einops import rearrange
from torch import nn

from common.data.data_point import EEGDatasetDataPoint
from common.model.embedding.embedder_adapter import EmbedderAdapter
from common.model.embedding.foundation_embedder import FoundationEmbedder
from common.model.embedding.predefined.cbramod import CBraModFoundationEmbedder
from common.model.embedding.predefined.minilm import MiniLMFoundationEmbedder
from common.model.embedding.predefined.vivit import ViViTFoundationEmbedder, ViViTFoundationEmbedderForTimeSequences
from common.model.embedding.predefined.w2vbert import W2VBertFoundationEmbedder
from common.model.layers.ISAB import ISAB, PMA
from common.model.layers.attention.x_attention import GatedCrossAttentionBlock

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


def media_locs_single_item(B, Tq, device):
    m = torch.zeros(B, Tq, dtype=torch.bool, device=device)
    m[:, 0] = True  # Item “introduced” at t=0
    return m


class SingleMediaEEGAVI(nn.Module):
    def __init__(self,
                 target_shape: int = 384,
                 cross_attention_blocks: int = 2,
                 base_video: FoundationEmbedder = ViViTFoundationEmbedderForTimeSequences(),
                 video_kd_size: int | None = None,

                 base_audio: FoundationEmbedder = W2VBertFoundationEmbedder(),
                 audio_kd_size: int | None = None,

                 base_text: FoundationEmbedder = MiniLMFoundationEmbedder(),
                 text_kd_size: int | None = None,
                 base_eeg: FoundationEmbedder = CBraModFoundationEmbedder(),
                 use_kd: bool = True):
        super().__init__()
        self.use_kd: bool = use_kd

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
            ),
        )

        self.text_adapter = EmbedderAdapter(
            embedder=base_text, target_size=target_shape, kd_size=text_kd_size,
            adapter=nn.Sequential(
                PMA(base_text.output_size, 8, 10)
            ),
        )

        self.eeg_adapter = EmbedderAdapter(
            embedder=base_eeg, target_size=target_shape,
            # Nothing happens for EEG data
            adapter=nn.Sequential(),
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

    def forward(self, x: dict | EEGDatasetDataPoint):
        # Order is unique
        order = list(zip(*x["order"]))[0]
        # By default, don't use KD
        use_kd = x["kd"] if "kd" in x else False
        mask = x["mask"] if "mask" in x else None
        # Access the stored data:
        eeg = x["eeg"]

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
        # TODO: Vediamo questo poi
        z_eeg = rearrange(z_eeg, "b c T D -> b (T c) D")
        media_locations = media_locs_single_item(b, T, z_eeg.device)

        # TODO new mask?
        embeddings = torch.cat(z_mod, dim=1)
        if len(embeddings.shape) == 3:
            # (b, T*F, D) (Case of no time series used).
            embeddings = rearrange(embeddings, "b T d -> b T F d", T=1)

        for gated_x_attn in self.gatedXAttn_layers:
            z_eeg = gated_x_attn(z_eeg, embeddings, media_locations=media_locations)

        # TODO: Self attention?

        logits = self.projector(z_eeg)
        return (logits, {"kd_ve": z_kd_vid, "kd_ae": z_kd_aud}) if use_kd else logits

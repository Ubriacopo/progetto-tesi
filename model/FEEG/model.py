from typing import Optional

import torch
from einops import rearrange
from torch import nn, Tensor

from core_data.data_point import EEGDatasetDataPoint
from common.model.embedding.predefined.cbramod import CBraModFoundationEmbedder
from common.model.embedding.predefined.minilm import MiniLMFoundationEmbedder
from common.model.embedding.predefined.w2vbert import W2VBertFoundationEmbedder
from common.model.embedding.predefined.vivit import ViViTFoundationEmbedder
from common.model.embedding.foundation_embedder import FoundationEmbedder
from model.layer.base import ModalContextEncoder
from common.model.embedding.embedder_adapter import EmbedderAdapter
from model.layer.attention.x_attention import GatedCrossAttentionBlock
from model.layer.ISAB import ISAB, PMA
from common.model.layers import PerceiverAdapter
from model.EEGAVI.transforms import media_locs_single_item

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


def freeze_module(m: torch.nn.Module):
    for p in m.parameters():
        p.requires_grad = False


class SimpleEEGAVI(nn.Module):
    def __init__(self,
                 target_shape: int = 384,
                 cross_attention_blocks: int = 2,
                 base_video: FoundationEmbedder = ViViTFoundationEmbedder(),
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
        order = list(zip(*x["order"]))[0]
        # By default, don't use KD
        use_kd = x["kd"] if "kd" in x else False

        # Access the stored data:
        eeg = x["eeg"]
        vid = x["vid"] if "vid" in x and x["mask"][0][order.index("vid")] else None
        aud = x["aud"] if "aud" in x and x["mask"][0][order.index("aud")] else None
        txt = x["txt"] if "txt" in x and x["mask"][0][order.index("txt")] else None

        kd_zv = None
        zv = self.video_adapter(vid, use_kd)
        if isinstance(zv, tuple):
            zv, kd_zv = zv

        kd_za = None
        za = self.audio_adapter(aud, use_kd)
        if isinstance(za, tuple):
            za, kd_za = za

        kd_zt = None
        zt = self.text_adapter(txt, use_kd)
        if isinstance(zt, tuple):
            zt, kd_zt = zt

        ze = self.eeg_adapter(eeg, use_kd)

        b, c, T, D = ze.shape
        ze = rearrange(ze, "b c T D -> b (T c) D")
        media_locations = media_locs_single_item(b, T, ze.device)
        embeddings = torch.cat([emb for emb in [zv, za, zt] if emb is not None], dim=1)

        if len(embeddings.shape) == 3:
            # (b, T*F, D) (Case of no time series used).
            embeddings = rearrange(embeddings, "b (T F) d -> b T F d", T=1)

        for gated_x_attn in self.gatedXAttn_layers:
            ze = gated_x_attn(ze, embeddings, media_locations=media_locations)

        logits = self.projector(ze)
        # Final projection head?
        return (logits, {"kd_ve": kd_zv, "kd_ae": kd_za, "kd_te": kd_zt}) if use_kd else logits


class ComplexEEGAVI(nn.Module):
    def __init__(self,
                 resampler_depth: int,
                 target_shape: int = 384,
                 cross_attention_blocks: int = 2,
                 base_video: FoundationEmbedder = ViViTFoundationEmbedder(),
                 video_kd_size: int | None = None,

                 base_audio: FoundationEmbedder = W2VBertFoundationEmbedder(),
                 audio_kd_size: int | None = None,

                 base_text: FoundationEmbedder = MiniLMFoundationEmbedder(),
                 text_kd_size: int | None = None,

                 base_eeg: FoundationEmbedder = CBraModFoundationEmbedder(),
                 use_kd: bool = True):
        super().__init__()

        self.use_kd: bool = use_kd
        # TODO Sostituire con ISAB+PMA
        # Free embedders = no ISAB needed. Frozen embedders = ISAB can be a useful adapter for long patchy sequences.
        self.video_adapter = PerceiverAdapter(base_video, resampler_depth, target_shape, use_kd, video_kd_size)
        # TODO Sostituire con PMA
        self.audio_adapter = PerceiverAdapter(base_audio, resampler_depth, target_shape, use_kd, audio_kd_size)

        # TODO Sostituire con PMA
        self.text_adapter = PerceiverAdapter(base_text, resampler_depth, target_shape, use_kd, text_kd_size)
        self.modal_encoder = ModalContextEncoder(target_shape, {"video": 0, "audio": 1, "text": 2})

        eeg_out: int = base_eeg.output_size
        self.base_eeg = base_eeg
        self.eeg_shape_adapter: Optional[nn.Sequential] = None
        if base_eeg.output_size != target_shape:
            self.eeg_shape_adapter = nn.Sequential(nn.LayerNorm(eeg_out), nn.Linear(eeg_out, target_shape))

        # I thought it to be a good idea but: CBraMod already encodes spatial and temporal information! No need to redo it.
        # self.eeg_aux_encoder = AuxiliaryEEGEncoder(target_shape, 5, 17)

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

    def forward(self, x, use_kd: bool = True):
        # As it should be we suppose the data to be processed for each encoder call.
        eeg, video, audio, text = x

        # Encode the info
        adapted_video = self.video_adapter(video, use_kd=use_kd)
        ve: Optional[torch.Tensor] = None if adapted_video is None else adapted_video[0]  # ve -> R_ve
        kd_ve: Optional[torch.Tensor] = None if adapted_video is None or not use_kd else adapted_video[1]
        ve = self.modal_encoder(ve, modality="video")

        adapted_audio = self.audio_adapter(audio, use_kd=use_kd)
        ae: Optional[torch.Tensor] = None if adapted_audio is None else adapted_audio[0]
        kd_ae: Optional[torch.Tensor] = None if adapted_audio is None or not use_kd else adapted_audio[1]
        ae = self.modal_encoder(ae, modality="audio")

        adapted_text = self.text_adapter(text, use_kd=use_kd)
        te: Optional[torch.Tensor] = None if adapted_text is None else adapted_text[0]
        kd_te: Optional[torch.Tensor] = None if adapted_text is None or not use_kd else adapted_text[1]
        te = self.modal_encoder(te, modality="text")

        ee: Tensor = self.base_eeg(**eeg, for_perceiver=False)
        if self.eeg_shape_adapter is not None:
            ee = self.eeg_shape_adapter(ee)

        # ee = self.eeg_aux_encoder(ee)
        # Given CBraMod’s design, adding LightEEGQueries with new time/channel embeddings is unnecessary and can hurt.
        b, c, T, D = ee.shape
        ee = rearrange(ee, "b c T D -> b (T c) D")

        # TODO: Non sono mica sicuro sia corretto
        # Works only because at the moment we have only 1 media at each timestep.
        # If we had multiple it'd be different.
        media_locations = media_locs_single_item(b, T, ee.device)
        # For gated attn
        embeddings = torch.cat([x for x in [ve, ae, te] if x is not None], dim=1)
        # Now we do Cross-Attention + Gating

        for gated_x_attn in self.gatedXAttn_layers:
            ee = gated_x_attn(ee, embeddings, media_locations=media_locations)

        logits = self.projector(ee)
        # Final projection head?
        return logits, {"kd_ve": kd_ve, "kd_ae": kd_ae, "kd_te": kd_te}


class EEGAVIMAG(nn.Module):
    def __init__(self):
        # self.merger = MAG3D(video_emb_size, y_dim=audio_emb_size, z_dim=text_emb_size, beta_shift=0.01, dropout=0)
        # We anchor EEG data now
        # self.eeg_merger = MAG2D(eeg_emb_size, y_dim=video_emb_size, beta_shift=0.01, dropout=0)
        super(EEGAVIMAG, self).__init__()

from typing import Optional

import torch
from einops import rearrange
from torch import nn, Tensor

from common.data.data_point import EEGModalityComposeWrapper, EEGDatasetDataPoint
from models.FEEG.layers.base_embedding import FoundationEmbedder, ViViTFoundationEmbedder, W2VBertFoundationEmbedder, \
    MiniLMFoundationEmbedder, CBraModFoundationEmbedder
from models.FEEG.layers.base_layers import ModalContextEncoder
from models.FEEG.layers.cross_attention import GatedCrossAttentionBlock
from models.FEEG.layers.isab import ISAB, PMA
from models.FEEG.layers.kd import KDHead
from models.FEEG.layers.perceiver_adapter import PerceiverAdapter

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


def media_locs_single_item(B, Tq, device):
    m = torch.zeros(B, Tq, dtype=torch.bool, device=device)
    m[:, 0] = True  # Item “introduced” at t=0
    return m


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
        self.use_kd: bool = use_kd
        self.video_adapter = nn.Sequential(
            ISAB(base_video.output_size, 8, 10),
            PMA(target_shape, 8, 10)
        )

        # KD
        if video_kd_size is not None:
            self.vid_kd_head = KDHead(base_video.output_size, video_kd_size)

        self.audio_adapter = nn.Sequential(
            PMA(target_shape, 8, 10)
        )

        if audio_kd_size is not None:
            self.aud_kd_head = KDHead(base_audio.output_size, audio_kd_size)

        self.text_adapter = nn.Sequential(
            PMA(target_shape, 8, 10)
        )

        if text_kd_size is not None:
            self.txt_kd_head = KDHead(base_text.output_size, text_kd_size)

    def forward(self, x: tuple[EEGDatasetDataPoint, bool] | EEGDatasetDataPoint):
        use_kd = False  # By default don't use KD
        if isinstance(x, tuple) and len(x) == 2:
            x, use_kd = x

        ve: Optional[torch.Tensor] = None
        kd_ve: Optional[torch.Tensor] = None
        if x.vid is not None:
            ve = self.video_adapter(x.vid, use_kd=use_kd)
            if use_kd and self.vid_kd_head is not None:
                kd_ve: Optional[torch.Tensor] = self.vid_kd_head(ve)

        ae = None
        kd_ae = None
        if x.aud is not None:
            ae = self.audio_adapter(x.aud, use_kd=use_kd)
            if use_kd and self.aud_kd_head is not None:
                kd_ae: Optional[torch.Tensor] = self.aud_kd_head(ae)

        te = None
        kd_te = None
        if x.txt is not None:
            te = self.text_adapter(x.txt, use_kd=use_kd)
            if use_kd and self.txt_kd_head is not None:
                kd_te: Optional[torch.Tensor] = self.txt_kd_head(te)

        ee = self.base_eeg(x.eeg, for_perceiver=False)
        b, c, T, D = ee.shape
        ee = rearrange(ee, "b c T D -> b (T c) D")
        media_locations = media_locs_single_item(b, T, ee.device)
        embeddings = torch.cat([e for e in [ve, ae, te] if e is not None], dim=1)

        for gated_x_attn in self.gatedXAttn_layers:
            ee = gated_x_attn(ee, embeddings, media_locations=media_locations)

        logits = self.projector(ee)
        # Final projection head?
        return (logits, {"kd_ve": kd_ve, "kd_ae": kd_ae, "kd_te": kd_te}) if use_kd else logits


class EEGAVI(nn.Module):
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

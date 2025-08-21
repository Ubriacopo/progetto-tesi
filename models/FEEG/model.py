from __future__ import annotations

from typing import Optional

import torch
from torch import nn, Tensor

from models.FEEG.base_embedding import FoundationEmbedder, ViViTFoundationEmbedder, W2VBertFoundationEmbedder, \
    MiniLMFoundationEmbedder, CBraModFoundationEmbedder
from models.FEEG.layers import KDHead, PerceiverResampler, GatedCrossAttentionBlock


def mask_last_preceding(q_time, Tm):  # q_time: (B,Tq) ints in [0..Tm]
    j = torch.arange(1, Tm + 1, device=q_time.device).view(1, 1, Tm)  # 1..Tm
    return (j == q_time.unsqueeze(-1))  # (B,Tq,Tm) booleans


def media_locations_from_q_time(q_time):
    # mark steps where the number of available chunks increases
    inc = torch.zeros_like(q_time, dtype=torch.bool)
    inc[:, 0] = q_time[:, 0] > 0
    inc[:, 1:] = q_time[:, 1:] > q_time[:, :-1]
    return inc  # (B, Tq)


class EEGAVI(torch.nn.Module):
    def __init__(self,
                 resampler_depth: int,
                 base_video: FoundationEmbedder = ViViTFoundationEmbedder(),
                 video_kd_size: int | None = None,

                 base_audio: FoundationEmbedder = W2VBertFoundationEmbedder(),
                 audio_kd_size: int | None = None,

                 base_text: FoundationEmbedder = MiniLMFoundationEmbedder(),
                 text_kd_size: int | None = None,

                 base_eeg: FoundationEmbedder = CBraModFoundationEmbedder(),
                 use_kd: bool = True):
        super(EEGAVI, self).__init__()
        self.use_kd: bool = use_kd
        target_shape = base_video.output_size
        # Video stuff: Todo: Turn into nn.Module?
        mod = self.build_modality(base_video, resampler_depth, target_shape, True, video_kd_size)
        self.base_video: FoundationEmbedder = mod[0]
        self.video_resampler: PerceiverResampler = mod[1]
        self.video_kd_head: Optional[KDHead] = mod[2]
        self.video_resampler_reshaper: torch.Linear = mod[3]

        # Audio stuff:
        mod = self.build_modality(base_audio, resampler_depth, target_shape, False, audio_kd_size)
        self.base_audio: FoundationEmbedder = mod[0]
        self.audio_resampler: PerceiverResampler = mod[1]
        self.audio_kd_head: Optional[KDHead] = mod[2]
        self.audio_resampler_reshaper: torch.Linear = mod[3]

        mod = self.build_modality(base_text, resampler_depth, target_shape, False, text_kd_size)
        self.base_text: FoundationEmbedder = mod[0]
        self.text_resampler: PerceiverResampler = mod[1]
        self.text_kd_head: Optional[KDHead] = mod[2]
        self.text_resampler_reshaper: torch.Linear = mod[3]

        self.base_eeg = base_eeg
        self.gatedXAttn = GatedCrossAttentionBlock(dim=base_eeg.output_size, dim_latent=self.base_video.output_size, )
        # TODO Revise this and choose a good architecture
        self.projector = nn.Sequential()

    # todo add layer for projection to correct space for mismatch on output size
    def build_modality(self, embedder: FoundationEmbedder, resampler_depth: int,
                       target_shape: int, kd: bool = False, kd_size: int = None) -> \
            (tuple[FoundationEmbedder, PerceiverResampler, KDHead | None, nn.Linear | None]):
        """

        :param target_shape:
        :param embedder:
        :param resampler_depth: How many layers of perceiver attention + sequential are desired for the PerceiverResampler
        :param kd: If the current modality uses knowledge distillation
        :param kd_size: To what size to remap the output of the Resampler to match the teacher
        :return:
        """
        resampler = PerceiverResampler(embedder.output_size, resampler_depth)
        kd_head: Optional[KDHead] = None
        if self.use_kd and kd:
            # Build KD map
            assert kd_size is not None, "If using KD you should provide kd_size to map to teacher"
            kd_head = KDHead(input_dimension=embedder.output_size, output_dimension=kd_size)

        reshaper: Optional[nn.Linear] = None
        if target_shape != embedder.output_size:
            reshaper = nn.Linear(embedder.output_size, target_shape)

        return embedder, resampler, kd_head, reshaper

    def forward(self, x):
        # As it should be we suppose the data to be processed for each encoder call.
        eeg, video, audio, text = x

        # Encode the info
        ve: Optional[torch.Tensor] = None  # ve -> R_ve
        kd_ve: Optional[torch.Tensor] = None
        if self.base_video is not None and video is not None:
            ve = self.base_video(**video)
            ve = self.video_resampler(ve)

            if self.video_kd_head is not None and self.use_kd:
                kd_ve = self.video_kd_head(ve)

            if self.video_resampler_reshaper is not None:
                ve = self.video_resampler_reshaper(ve)

        ae: Optional[torch.Tensor] = None
        kd_ae: Optional[torch.Tensor] = None
        if self.base_audio is not None and audio is not None:
            ae = self.base_audio(**audio)
            ae = self.audio_resampler(ae)

            if self.audio_kd_head is not None and self.use_kd:
                kd_ae = self.audio_kd_head(ae)

            if self.audio_resampler_reshaper is not None:
                ae = self.audio_resampler_reshaper(ae)

        te: Optional[torch.Tensor] = None
        kd_te: Optional[torch.Tensor] = None
        if self.base_text is not None and text is not None:
            te = self.base_text(**text)
            te = self.text_resampler(te)

            if self.text_kd_head is not None and self.use_kd:
                kd_te = self.text_kd_head(te)

            if self.text_resampler_reshaper is not None:
                te = self.text_resampler_reshaper(te)

        ee: Optional[Tensor] = self.base_eeg(for_perceiver=False, **eeg)
        # For gated attn
        embeddings = torch.cat([x for x in [ve, ae, te] if x is not None], dim=1)

        # TODO: la gated attn non ha shape fissa?
        # Now we do Cross-Attention + Gating
        ee = self.gatedXAttn(ee, embeddings)
        logits = self.projector(ee)
        # Final projection head?
        return logits, {"kd_ve": kd_ve, "kd_ae": kd_ae, "kd_te": kd_te}


class EEGAVIMAG(torch.nn.Module):
    def __init__(self):
        # self.merger = MAG3D(video_emb_size, y_dim=audio_emb_size, z_dim=text_emb_size, beta_shift=0.01, dropout=0)
        # We anchor EEG data now
        # self.eeg_merger = MAG2D(eeg_emb_size, y_dim=video_emb_size, beta_shift=0.01, dropout=0)
        super(EEGAVIMAG, self).__init__()

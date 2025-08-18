from __future__ import annotations

import torch
from torch import nn, Tensor

from models.FEEG.base_embedding import BaseEmbedding
from models.FEEG.layers import KDHead
from models.FEEG.mag import MAG3D, MAG2D


class EEGAVI(torch.nn.Module):
    def __init__(self,
                 base_video: BaseEmbedding = BaseEmbedding.get_ViViT_base(),
                 video_kd_size: int | None = None,

                 base_audio: BaseEmbedding = BaseEmbedding.get_wav2vec_base(),
                 audio_kd_size: int | None = None,

                 base_text: BaseEmbedding = BaseEmbedding.get_BERT_base(),
                 text_kd_size: int | None = None,

                 base_eeg: BaseEmbedding = BaseEmbedding.get_eeg_former_base(),
                 use_kd: bool = True):
        super(EEGAVI).__init__()
        self.use_kd: bool = use_kd
        self.base_video = base_video

        self.video_kd_head: KDHead | None = None
        video_emb_size = self.base_video.output_size
        if video_kd_size is not None:
            self.video_kd_head = KDHead(input_dimension=video_emb_size, output_dimension=video_kd_size)

        self.base_audio = base_audio
        audio_emb_size = self.base_audio.output_size
        self.audio_kd_head: KDHead | None = None
        if audio_kd_size is not None:
            self.audio_kd_head = KDHead(input_dimension=self.base_audio.output_size, output_dimension=audio_kd_size)

        self.base_text = base_text
        text_emb_size = self.base_text.output_size
        self.text_kd_head: KDHead | None = None
        if text_kd_size is not None:
            self.text_kd_head = KDHead(input_dimension=self.base_text.output_size, output_dimension=text_kd_size)

        self.base_eeg = base_eeg
        eeg_emb_size = self.base_eeg.output_size
        # We anchor video.
        self.merger = MAG3D(video_emb_size, y_dim=audio_emb_size, z_dim=text_emb_size, beta_shift=0.01, dropout=0)
        # We anchor EEG data now
        self.eeg_merger = MAG2D(eeg_emb_size, y_dim=video_emb_size, beta_shift=0.01, dropout=0)

        self.projector = nn.Sequential()

    def call_encoder(self, x, embedder: nn.Module | BaseEmbedding, kd_head: KDHead | None = None,
                     ignore_kd: bool = False) -> tuple[None | Tensor, None | Tensor] | None | Tensor:
        if x is None:
            return None

        x = embedder(x)
        if isinstance(x, BaseEmbedding):
            x = embedder.retrieve_logits(x)

        kd_x = None
        if kd_head is not None and self.use_kd:
            kd_x = kd_head(x)
        return x if not self.use_kd or ignore_kd else x, kd_x

    def forward(self, x):
        eeg, video, audio, text = x
        # Encode the info
        ve, kd_ve = self.call_encoder(video, embedder=self.base_video, kd_head=self.video_kd_head)
        ae, kd_ae = self.call_encoder(audio, embedder=self.base_audio, kd_head=self.audio_kd_head)
        te, kd_te = self.call_encoder(text, embedder=self.base_text, kd_head=self.text_kd_head)
        ee: None | Tensor = self.call_encoder(eeg, embedder=self.base_eeg, ignore_kd=True)

        # Merge embeddings (video, text, audio)
        # TODO: Handle None values in merger
        avt_embeddings = self.merger(anchor=ve, y=ae, z=te)
        # Merge embeddings (m1 eeg)
        avt_eeg_embeddings = self.eeg_merger(anchor=ee, y=avt_embeddings)
        logits = self.projector(avt_eeg_embeddings)

        # Final projection head?
        return logits, {"kd_ve": kd_ve, "kd_ae": kd_ae, "kd_te": kd_te}

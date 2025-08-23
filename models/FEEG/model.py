from typing import Optional

import torch
from torch import nn, Tensor

from models.FEEG.layers.base_embedding import FoundationEmbedder, ViViTFoundationEmbedder, W2VBertFoundationEmbedder, \
    MiniLMFoundationEmbedder, CBraModFoundationEmbedder
from models.FEEG.layers.base_layers import AuxiliaryEEGEncoder, ModalContextEncoder
from models.FEEG.layers.cross_attention import GatedCrossAttentionBlock
from models.FEEG.layers.perceiver_adapter import PerceiverAdapter


class EEGAVI(nn.Module):
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
        super().__init__()
        self.use_kd: bool = use_kd
        target_shape = base_video.output_size
        self.video_adapter = PerceiverAdapter(base_video, resampler_depth, target_shape, True, video_kd_size)
        self.audio_adapter = PerceiverAdapter(base_audio, resampler_depth, target_shape, True, audio_kd_size)
        self.text_adapter = PerceiverAdapter(base_text, resampler_depth, target_shape, True, text_kd_size)

        self.modal_encoder = ModalContextEncoder(base_video.output_size, 3, {"video": 0, "audio": 1, "text": 2})

        self.base_eeg = base_eeg

        # Todo: pass for now hardwired
        self.aux_eeg_encoder = AuxiliaryEEGEncoder(base_eeg.output_size, 5, 17)
        self.gatedXAttn = GatedCrossAttentionBlock(dim=base_eeg.output_size, dim_latent=base_video.output_size, )

        # TODO Revise this and choose a good architecture
        self.projector = nn.Sequential()

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

        ee: Tensor = self.base_eeg(for_perceiver=False, **eeg)
        ee = self.aux_eeg_encoder(ee)

        # For gated attn
        embeddings = torch.cat([x for x in [ve, ae, te] if x is not None], dim=1)
        # Now we do Cross-Attention + Gating
        ee = self.gatedXAttn(ee, embeddings)
        logits = self.projector(ee)
        # Final projection head?
        return logits, {"kd_ve": kd_ve, "kd_ae": kd_ae, "kd_te": kd_te}


class EEGAVIMAG(nn.Module):
    def __init__(self):
        # self.merger = MAG3D(video_emb_size, y_dim=audio_emb_size, z_dim=text_emb_size, beta_shift=0.01, dropout=0)
        # We anchor EEG data now
        # self.eeg_merger = MAG2D(eeg_emb_size, y_dim=video_emb_size, beta_shift=0.01, dropout=0)
        super(EEGAVIMAG, self).__init__()

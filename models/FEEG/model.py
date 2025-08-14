from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable

import torch
from torch import nn
from transformers import VivitModel, Wav2Vec2FeatureExtractor, BertModel


class BaseEmbedding(ABC):
    def __init__(self, model, output_size: int):
        self.model = model
        self.output_size = output_size

    @abstractmethod
    def retrieve_logits(self, x):
        pass


class LambdaBaseEmbedding(BaseEmbedding):
    def __init__(self, model, output_size: int, retrieve_logits: Callable[[torch.Tensor], torch.Tensor]):
        super().__init__(model, output_size)
        self.lambda_fn = retrieve_logits

    def retrieve_logits(self, x):
        return self.lambda_fn(x)


class ViVitBaseEmbedding(BaseEmbedding):
    # Simple showcase
    def __init__(self):
        super().__init__(VivitModel.from_pretrained("google/vivit-b-16x2-kinetics400"), 768)

    def retrieve_logits(self, x):
        return x.last_hidden_state


class Wav2Vec2BaseEmbedding(BaseEmbedding):
    def __init__(self):
        # todo vedi size effettiva
        super().__init__((Wav2Vec2FeatureExtractor.from_pretrained("facebook/wav2vec2-base-960h")), 768)

    def retrieve_logits(self, x):
        return x.logits


class EEGAVI(torch.nn.Module):
    @staticmethod
    def build() -> EEGAVI:
        bert = BertModel.from_pretrained("google/electra-base-discriminator")
        return EEGAVI(
            base_video=ViVitBaseEmbedding(),
            base_audio=Wav2Vec2BaseEmbedding(),
            # todo finish.
            base_text=LambdaBaseEmbedding(bert, 768, lambda x: x),
            base_eeg=LambdaBaseEmbedding(),
        )

    def __init__(self, base_video: BaseEmbedding, base_audio: BaseEmbedding,
                 base_text: BaseEmbedding, base_eeg: BaseEmbedding):
        super(EEGAVI).__init__()

        self.base_video = base_video
        assert self.base_video.output_size > 512, "We cannot condense information if the downstream item is bigger"
        self.video_encoder = nn.Sequential(
            nn.Linear(self.base_video.output_size, 512),
            nn.ReLU(),  # Activation fn
            nn.Linear(512, 256),
        )

        self.base_audio = base_audio
        assert self.base_audio.output_size > 384, "We cannot condense information if the downstream item is smaller"
        self.audio_encoder = nn.Sequential(
            nn.Linear(self.base_audio.output_size, 384),
            nn.ReLU(),  # Activation fn
            nn.Linear(384, 256),
        )

        self.base_text = base_text
        assert self.base_text.output_size > 384, "We cannot condense information if the downstream item is smaller"
        self.text_encoder = nn.Sequential(
            nn.Linear(self.base_text.output_size, 384),
            nn.ReLU(),  # Activation fn
            nn.Linear(384, 256),
        )

        self.base_eeg = base_eeg
        assert self.base_eeg.output_size > 384, "We cannot condense information if the downstream item is smaller"
        self.eeg_encoder = nn.Sequential(
            nn.Linear(self.base_eeg.output_size, 384),
            nn.ReLU(),
            nn.Linear(384, 320),
            nn.ReLU(),
            nn.Linear(320, 256),
        )

    @staticmethod
    def call_encoder(x, embedder: nn.Module | BaseEmbedding, encoder: nn.Module):
        if x is None:
            return None

        x = embedder(x)
        if isinstance(x, BaseEmbedding):
            x = embedder.retrieve_logits(x)

        x = encoder(x)
        return x

    def forward(self, x):
        eeg, video, audio, text = x

        ve = EEGAVI.call_encoder(video, embedder=self.base_video, encoder=self.video_encoder)
        ae = EEGAVI.call_encoder(audio, embedder=self.base_audio, encoder=self.audio_encoder)
        te = EEGAVI.call_encoder(text, embedder=self.base_text, encoder=self.text_encoder)
        ee = EEGAVI.call_encoder(eeg, embedder=self.base_eeg, encoder=self.eeg_encoder)

        # Merge embeddings (video, text, audio)


        # Merge embeddings (m1 eeg)

        # Final projection head? Nah ?


# For first approach we try using a pre-defined embedding model.
class FEEG(torch.nn.Module):
    def __init__(self, video_embedder, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.layers = []

        self.video_embedding_model: VivitModel = VivitModel.from_pretrained("google/vivit-b-16x2-kinetics400")
        # Structure it
        self.video_encoder = nn.Sequential(
            # Todo pensa a un meccanismo (vedi mufffin) (1d conv?) (tranformer?)
            nn.Linear(768, 400),
            nn.ReLU(),
            nn.Linear(400, 256),
        )  # ViViT

        self.audio_encoder = None  # HuBERT

        self.text_encoder = None  # BERT or MiniLM
        # For EEG none (We distill from a model)
        self.eeg_encoder = None

        # TODO modellalo
        self.encoder = nn.Sequential(
            nn.Linear(768 * 2, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
        )

    def forward(self, x):
        eeg, video, audio, text = x
        eeg_embed, video_embed, audio_embed, text_embed = None, None, None, None

        if eeg is not None:
            eeg_embed = self.eeg_encoder(eeg)
        if video is not None:
            video_embed = self.video_encoder(video)
        if audio is not None:
            audio_embed = self.audio_encoder(audio)
        if text is not None:
            text_embed = self.text_encoder(text)

        # Merge the embeddings. (Concat input)
        embeddings = eeg_embed + video_embed + audio_embed + text_embed
        res = self.encoder(embeddings)
        # compute loss
        return res

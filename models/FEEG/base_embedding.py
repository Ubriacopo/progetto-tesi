from abc import abstractmethod, ABC

import torch
from cbramod.models.cbramod import CBraMod
from torch import nn
from transformers import VivitModel, WavLMModel, AutoModel, Wav2Vec2BertModel


class FoundationEmbedder(nn.Module, ABC):
    def __init__(self, base_model, output_size: int):
        super().__init__()
        self.base_model = base_model
        self.output_size: int = output_size

    @abstractmethod
    def retrieve_logits(self, x):
        raise NotImplementedError

    @abstractmethod
    def reshape_for_perceiver(self, x):
        raise NotImplementedError

    def forward(self, for_perceiver: bool = True, *args, **kwargs) -> torch.Tensor:
        x = self.base_model(*args, **kwargs)
        x = self.retrieve_logits(x)
        return self.reshape_for_perceiver(x) if for_perceiver else x


class ViViTFoundationEmbedder(FoundationEmbedder):
    def __init__(self, output_size: int = 768, variant: str = "google/vivit-b-16x2-kinetics400"):
        super().__init__(VivitModel.from_pretrained(variant), output_size)

    def reshape_for_perceiver(self, x):
        tokens = x[:, 1:, :]  # Drop the [CLS] token
        b, N, D = tokens.shape  # Shape given by ViViT
        assert N % self.base_model.config.num_frames == 0, \
            f"Token count {N} is not divisible by self.frames={self.model.config.num_frames}. " \
            "Check that self.frames matches ViViT's config.num_frames."

        v = int(N / self.base_model.config.num_frames)  # Num patches for frame
        return tokens.reshape(b, self.base_model.config.num_frames, 1, v, D)

    def retrieve_logits(self, x):
        return x.last_hidden_state


class W2VBertFoundationEmbedder(FoundationEmbedder):
    def __init__(self, output_size: int = 1024, variant: str = "facebook/w2v-bert-2.0"):
        super().__init__(Wav2Vec2BertModel.from_pretrained(variant), output_size)

    def retrieve_logits(self, x):
        return x.last_hidden_state

    def reshape_for_perceiver(self, x):
        b, N, D = x.shape
        return x.reshape(b, 1, N, 1, D)


class MiniLMFoundationEmbedder(FoundationEmbedder):
    def __init__(self, output_size: int = 384, variant="sentence-transformers/all-MiniLM-L6-v2",
                 perceiver_default: bool = True):
        super().__init__(AutoModel.from_pretrained(variant), output_size)

    def retrieve_logits(self, x):
        return x.last_hidden_state

    def reshape_for_perceiver(self, x):
        b, N, D = x.shape
        return x.reshape(b, 1, N, 1, D)


class CBraModFoundationEmbedder(FoundationEmbedder):

    def __init__(self, output_size: int = 200, device=None,
                 weights: str = "../../dependencies/cbramod/pretrained_weights.pth", perceiver_default: bool = True):
        model = CBraMod()
        if device is None:
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        model.load_state_dict(torch.load(weights, map_location=device))
        super().__init__(model, output_size)

    def retrieve_logits(self, x):
        return x

    def reshape_for_perceiver(self, x):
        return x

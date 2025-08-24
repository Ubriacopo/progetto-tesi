from abc import abstractmethod, ABC

import torch
from cbramod.models.cbramod import CBraMod
from torch import nn
from transformers import VivitModel, WavLMModel, AutoModel, Wav2Vec2BertModel

from models.FEEG.utils import freeze_module


class FoundationEmbedder(nn.Module, ABC):
    """
    Container class to handle calling a foundation model for embeddings.
    """

    def __init__(self, base_model, output_size: int, freeze: bool):
        super().__init__()
        if freeze:
            freeze_module(base_model)

        self.base_model = base_model
        self.output_size: int = output_size

    @abstractmethod
    def retrieve_patches(self, x):
        raise NotImplementedError

    @abstractmethod
    def reshape_for_perceiver(self, x):
        raise NotImplementedError

    def forward(self, for_perceiver: bool = True, *args, **kwargs) -> torch.Tensor:
        x = self.base_model(*args, **kwargs)
        x = self.retrieve_patches(x)
        return self.reshape_for_perceiver(x) if for_perceiver else x


class ViViTFoundationEmbedder(FoundationEmbedder):
    def __init__(self, output_size: int = 768, variant: str = "google/vivit-b-16x2-kinetics400", freeze: bool = True):
        super().__init__(VivitModel.from_pretrained(variant), output_size, freeze)

    def reshape_for_perceiver(self, x):
        tokens = x[:, 1:, :]  # Drop the [CLS] token
        b, N, D = tokens.shape  # Shape given by ViViT
        assert N % self.base_model.config.num_frames == 0, \
            f"Token count {N} is not divisible by self.frames={self.model.config.num_frames}. " \
            "Check that self.frames matches ViViT's config.num_frames."

        v = int(N / self.base_model.config.num_frames)  # Num patches for frame
        return tokens.reshape(b, self.base_model.config.num_frames, 1, v, D)

    def retrieve_patches(self, x):
        return x.last_hidden_state


class W2VBertFoundationEmbedder(FoundationEmbedder):
    def __init__(self, output_size: int = 1024, variant: str = "facebook/w2v-bert-2.0", freeze: bool = True):
        super().__init__(Wav2Vec2BertModel.from_pretrained(variant), output_size, freeze)

    def retrieve_patches(self, x):
        return x.last_hidden_state

    def reshape_for_perceiver(self, x):
        b, N, D = x.shape
        return x.reshape(b, 1, N, 1, D)


class MiniLMFoundationEmbedder(FoundationEmbedder):
    def __init__(self, output_size: int = 384, variant="sentence-transformers/all-MiniLM-L6-v2", freeze: bool = True):
        super().__init__(AutoModel.from_pretrained(variant), output_size, freeze)

    def retrieve_patches(self, x):
        return x.last_hidden_state

    def reshape_for_perceiver(self, x):
        b, N, D = x.shape
        return x.reshape(b, 1, N, 1, D)


class CBraModFoundationEmbedder(FoundationEmbedder):

    def __init__(self, output_size: int = 200, device=None, freeze: bool = True,
                 weights: str = "../../dependencies/cbramod/pretrained_weights.pth"):
        model = CBraMod()
        if device is None:
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        assert weights is not None or freeze is False, "If you freeze the model has to have a state"
        if weights is not None:
            model.load_state_dict(torch.load(weights, map_location=device))

        super().__init__(model, output_size, freeze)

    def retrieve_patches(self, x):
        return x

    def reshape_for_perceiver(self, x):
        return x

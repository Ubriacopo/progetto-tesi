import torch
from einops import rearrange
from transformers import VivitModel

from common.model.embedding.foundation_embedder import FoundationEmbedder


# TODO Variante che supporta time segments
class ViViTFoundationEmbedder(FoundationEmbedder):
    def __init__(self, output_size: int = 768, variant: str = "google/vivit-b-16x2-kinetics400", freeze: bool = True):
        super().__init__(VivitModel.from_pretrained(variant), output_size, freeze)

    def reshape_for_perceiver(self, x):
        tokens = x[:, 1:, :]  # Drop the [CLS] token
        b, N, D = tokens.shape  # Shape given by ViViT
        tubelet = self.base_model.config.tubelet_size[-1]
        assert N % self.base_model.config.num_frames == 0, \
            f"Token count {N} is not divisible by self.frames={self.model.config.num_frames}. " \
            "Check that self.frames matches ViViT's config.num_frames."

        v = int(N / tubelet)  # Num patches for frame
        return tokens.reshape(b, 1, tubelet, v, D)

    def retrieve_patches(self, x):
        return x.last_hidden_state

# Todo ma mask?
class ViViTFoundationEmbedderForTimeSequences(FoundationEmbedder):
    def reshape_for_perceiver(self, x):
        raise NotImplementedError()

    def retrieve_patches(self, x):
        return x.last_hidden_state

    def __init__(self, output_size: int = 768, variant: str = "google/vivit-b-16x2-kinetics400", freeze: bool = True):
        super().__init__(VivitModel.from_pretrained(variant), output_size, freeze)

    def forward(self, x, for_perceiver: bool = False) -> torch.Tensor:
        b = x.pixel_values.shape[0]  # Batch size
        x.pixel_values = rearrange(x.pixel_values, "b T f c w h -> (b T) f c w h")

        if self.model_is_frozen:
            with torch.no_grad():
                x = self.base_model(x.pixel_values)
        else:
            x = self.base_model(x.pixel_values)

        x.last_hidden_state = rearrange(x.last_hidden_state, "(b T) F D -> b T F D", b=b)
        return self.reshape_for_perceiver(x) if for_perceiver else x

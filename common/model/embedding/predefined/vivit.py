import torch
from einops import rearrange
from transformers import VivitModel

from common.model.embedding.foundation_embedder import FoundationEmbedder


class ViViTFoundationEmbedder(FoundationEmbedder):
    def __init__(self, output_size: int = 768, variant: str = "google/vivit-b-16x2-kinetics400", freeze: bool = True):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        super().__init__(VivitModel.from_pretrained(variant, device_map=device), output_size, freeze)

    def forward(self, x, mask=None) -> torch.Tensor:
        # TODO: Masking
        x = x.to(self.base_model.device)
        x = x.pixel_values

        if len(x.shape) == 4:
            # Add a virtual batch
            x = x.unsqueeze(0)

        if self.model_is_frozen:
            with torch.no_grad():
                y = self.base_model(x)
        else:
            y = self.base_model(x)

        y = y.last_hidden_state  # Take the real values.
        # Drop the [CLS] token
        tokens = y[:, 1:, :]

        # tubelet = self.base_model.config.tubelet_size[-1]
        # tokens = rearrange(tokens, "b (F p) D -> b F p D", F=tubelet)

        return tokens

    def get_output_shape(self, b: int = -1) -> tuple[int, ...]:
        return b, self.base_model.config.tubelet_size[-1], int(3306 / self.base_model.config.tubelet_size[-1]), 768


class ViViTFoundationEmbedderForTimeSequences(FoundationEmbedder):
    def __init__(self, output_size: int = 768, variant: str = "google/vivit-b-16x2-kinetics400", freeze: bool = True):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        super().__init__(VivitModel.from_pretrained(variant, device_map=device), output_size, freeze)

    def forward(self, x, mask=None) -> torch.Tensor:
        # TODO: Masking
        b = x.pixel_values.shape[0]  # Batch size
        x = x.to(self.base_model.device)

        x.pixel_values = rearrange(x.pixel_values, "b T f c w h -> (b T) f c w h")
        if self.model_is_frozen:
            with torch.no_grad():
                x = self.base_model(x.pixel_values)
        else:
            x = self.base_model(x.pixel_values)

        y = rearrange(x.last_hidden_state, "(b T) F D -> b T F D", b=b)
        tokens = y[:, :, 1:, :]  # Drop the [CLS] token
        tubelet = self.base_model.config.tubelet_size[-1]
        tokens = rearrange(tokens, "b T (F p) D -> b T F p D", F=tubelet)
        return tokens

    def get_output_shape(self, b: int = -1, t: int = -1) -> tuple[int, ...]:
        return b, t, self.base_model.config.tubelet_size[-1], int(3306 / self.base_model.config.tubelet_size[-1]), 768

import torch
from einops import rearrange
from transformers import Wav2Vec2BertModel

from common.model.embedding.foundation_embedder import FoundationEmbedder


class W2VBertFoundationEmbedder(FoundationEmbedder):
    def __init__(self, output_size: int = 1024, variant: str = "facebook/w2v-bert-2.0", freeze: bool = True,
                 for_time_series: bool = False):
        super().__init__(Wav2Vec2BertModel.from_pretrained(variant), output_size, freeze)
        self.for_time_series = for_time_series

    def forward(self, x, mask=None) -> torch.Tensor:
        # TODO Usa mask
        b = x.input_features.shape[0]  # Batch size

        input_features = x.input_features
        attn_mask = x.attention_mask

        if self.for_time_series:
            input_features = rearrange(x.input_features, "b T p d -> (b T) p d")
            attn_mask = rearrange(x.attention_mask, "b T p -> (b T) p")

        if self.model_is_frozen:
            with torch.no_grad():
                y = self.base_model(input_features, attn_mask)
        else:
            y = self.base_model(input_features, attn_mask)
        y = y.last_hidden_state

        if self.for_time_series:
            y = rearrange(y, "(b T) F D -> b T F D", b=b)
        return y

    # todo Cambia un poco, non sono le dimensioni a importare ma solo quante sono? Pensaci
    def get_output_shape(self, b: int = -1, t: int = -1) -> tuple[int, ...]:
        return (b, t, 399, 1, 160) if self.for_time_series else (b, 399, 1, 160)

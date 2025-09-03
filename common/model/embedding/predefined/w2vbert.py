import torch
from einops import rearrange
from torch import nn
from transformers import Wav2Vec2BertModel

from common.model.embedding.foundation_embedder import FoundationEmbedder


class W2VBertFoundationEmbedder(FoundationEmbedder):
    def __init__(self, output_size: int = 1024, variant: str = "facebook/w2v-bert-2.0", freeze: bool = True):
        super().__init__(Wav2Vec2BertModel.from_pretrained(variant), output_size, freeze)

    def retrieve_patches(self, x):
        return x.last_hidden_state

    def reshape_for_perceiver(self, x):
        b, N, D = x.shape
        return x.reshape(b, 1, N, 1, D)


class W2VBertFoundationEmbedderForTimeSequences(FoundationEmbedder):
    def reshape_for_perceiver(self, x):
        pass

    def retrieve_patches(self, x):
        pass

    def __init__(self, output_size: int = 1024, variant: str = "facebook/w2v-bert-2.0", freeze: bool = True):
        super().__init__(Wav2Vec2BertModel.from_pretrained(variant), output_size, freeze)

    def forward(self, x, for_perceiver: bool = False) -> torch.Tensor:
        b = x.input_features.shape[0]  # Batch size
        input_features = rearrange(x.input_features, "b T p d -> (b T) p d")
        attn_mask = rearrange(x.attention_mask, "b T p -> (b T) p")

        if self.model_is_frozen:
            with torch.no_grad():
                x = self.base_model(input_features, attn_mask)
        else:
            x = self.base_model(input_features, attn_mask)

        y = rearrange(x.last_hidden_state, "(b T) F D -> b T F D", b=b)
        y = y.unsqueeze(2) # W2V has no other decomposition
        return y

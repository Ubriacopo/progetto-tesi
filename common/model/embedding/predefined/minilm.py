from transformers import AutoModel

from common.model.embedding.foundation_embedder import FoundationEmbedder


class MiniLMFoundationEmbedder(FoundationEmbedder):
    def __init__(self, output_size: int = 384, variant="sentence-transformers/all-MiniLM-L6-v2", freeze: bool = True):
        super().__init__(AutoModel.from_pretrained(variant), output_size, freeze)

    def retrieve_patches(self, x):
        return x.last_hidden_state

    def reshape_for_perceiver(self, x):
        b, N, D = x.shape
        return x.reshape(b, 1, N, 1, D)

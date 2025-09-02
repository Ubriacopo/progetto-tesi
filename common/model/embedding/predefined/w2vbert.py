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

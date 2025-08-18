from abc import abstractmethod, ABC
from typing import Callable

import torch
from transformers import VivitModel, Wav2Vec2FeatureExtractor, BertModel


class BaseEmbedding(ABC):
    def __init__(self, model, output_size: int):
        self.model = model
        self.output_size = output_size

    @abstractmethod
    def retrieve_logits(self, x):
        pass

    @staticmethod
    def get_ViViT_base():
        model = VivitModel.from_pretrained("google/vivit-b-16x2-kinetics400")
        return LambdaBaseEmbedding(model, 768, lambda x: x.last_hidden_state)

    @staticmethod
    def get_wav2vec_base():
        model = Wav2Vec2FeatureExtractor.from_pretrained("facebook/wav2vec2-base-960h")
        return LambdaBaseEmbedding(model, 768, lambda x: x.logits)

    @staticmethod
    def get_BERT_base():
        bert = BertModel.from_pretrained("google/electra-base-discriminator")
        return LambdaBaseEmbedding(bert, 768, lambda x: x)

    @staticmethod
    def get_eeg_former_base():
        eeg_model = None  # todo
        return LambdaBaseEmbedding(eeg_model, 768, lambda x: x)


class LambdaBaseEmbedding(BaseEmbedding):
    def __init__(self, model, output_size: int, retrieve_logits: Callable[[torch.Tensor], torch.Tensor]):
        super().__init__(model, output_size)
        self.lambda_fn = retrieve_logits

    def retrieve_logits(self, x):
        return self.lambda_fn(x)

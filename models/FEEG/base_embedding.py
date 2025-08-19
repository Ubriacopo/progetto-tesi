from abc import abstractmethod, ABC
from typing import Callable

import torch
from cbramod.models.cbramod import CBraMod
from transformers import VivitModel, Wav2Vec2FeatureExtractor, BertModel, WavLMModel


class BaseEmbedding(ABC):
    def __init__(self, model, output_size: int):
        self.model = model
        self.output_size = output_size

    @abstractmethod
    def retrieve_logits(self, x):
        pass

    @staticmethod
    def get_ViViT_base():
        # Wants 32 frames clips (so
        model = VivitModel.from_pretrained("google/vivit-b-16x2-kinetics400")
        return LambdaBaseEmbedding(model, 768, lambda x: x.last_hidden_state)

    @staticmethod
    def get_wav2vec_base():
        # Input must be raw waveform sampled at 16,000 Hz.
        # If your audio is at 44.1 kHz, 48 kHz, etc., you must resample before feeding it in.
        model = WavLMModel.from_pretrained("microsoft/wavlm-base")
        return LambdaBaseEmbedding(model, 768, lambda x: x.logits)

    @staticmethod
    def get_BERT_base():
        bert = BertModel.from_pretrained("google/electra-base-discriminator")
        return LambdaBaseEmbedding(bert, 768, lambda x: x)

    @staticmethod
    def get_cbramod_base(device=None, weights_path: str = "../../dependencies/cbramod/pretrained_weights.pth"):
        model = CBraMod()
        if device is None:
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        model.load_state_dict(torch.load(weights_path, map_location=device))
        return LambdaBaseEmbedding(model, 200, lambda x: x)

    def __call__(self, x):
        res = self.model(x)
        return self.retrieve_logits(res)


class LambdaBaseEmbedding(BaseEmbedding):
    def __init__(self, model, output_size: int, retrieve_logits: Callable[[torch.Tensor], torch.Tensor]):
        super().__init__(model, output_size)
        self.lambda_fn = retrieve_logits

    def retrieve_logits(self, x):
        return self.lambda_fn(x)

from __future__ import annotations

import torch
from transformers import BertModel, BertTokenizer

from .media import MediaPreProcessingPipeline, PandasCsvDataMediaCollector


class TextCollector(PandasCsvDataMediaCollector):
    @staticmethod
    def AMIGOS(processor: MediaPreProcessingPipeline):
        return TextCollector([], processor)


class TextProcessingPipeline(MediaPreProcessingPipeline):
    def process_output_shape(self) -> tuple:
        return ()  # todo

    @staticmethod
    def default() -> TextProcessingPipeline:
        return TextProcessingPipeline(
            BertTokenizer.from_pretrained("bert-base-uncased"),
            BertModel.from_pretrained("bert-base-uncased")
        )

    def __init__(self, tokenizer, model):
        self.tokenizer = tokenizer
        self.model = model

    def process(self, media: list | torch.Tensor | str):
        embeddings = torch.tensor([self.tokenizer.encode(str(media))])
        with torch.no_grad():
            return self.model(embeddings).pooler_output.squeeze(0)

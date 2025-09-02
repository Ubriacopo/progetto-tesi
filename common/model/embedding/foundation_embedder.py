from abc import ABC, abstractmethod

import torch
from torch import nn

from common.model.utils import freeze_module

# TODO Drop the facade. Almeno cambia contratto. Forward sarà da definire a manina per ognuno
#               Più facile da usare così poi
class FoundationEmbedder(nn.Module, ABC):
    def __init__(self, base_model, output_size: int, freeze: bool):
        """
        Container class to handle calling a foundation model for embeddings.
        :param base_model: Base foundation model.
        :param output_size: Output size to remap (if necessary) the embeddings.
        :param freeze: If true, the base model is frozen.
        """
        super().__init__()
        self.model_is_frozen: bool = freeze
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

    def forward(self, x, for_perceiver: bool = False) -> torch.Tensor:
        if self.model_is_frozen:
            with torch.no_grad():
                x = self.base_model(**x)
        else:
            x = self.base_model(**x)

        x = self.retrieve_patches(x)
        return self.reshape_for_perceiver(x) if for_perceiver else x

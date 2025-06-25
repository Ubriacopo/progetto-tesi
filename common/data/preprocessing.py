from abc import ABC, abstractmethod

import numpy as np
import torch


class MediaPreProcessingPipeline(ABC):
    @abstractmethod
    def process(self, media: list | np.ndarray | str) -> np.ndarray | torch.Tensor:
        pass

    @abstractmethod
    def process_output_shape(self) -> tuple:
        pass

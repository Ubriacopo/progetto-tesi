from typing import Tuple

import torch
from torch import nn

# todo abstract
class KDHead(nn.Module):
    def __init__(self, input_size: int, target_shape: Tuple[int, ...], ):
        super(KDHead, self).__init__()
        # TODO: Evaluate as said.
        # Output shape is teacher shape
        # Practical advice
        #   Start linear only: Linear(768→384) → Linear(384→100).
        #   If you see KD loss stuck high, embeddings not aligning, or gradients vanishing, add:
        #   Linear(768→384), GELU, LayerNorm(384), Linear(384→100)
        # For safety: you can make the non-linearity optional via a flag in your KDHead, so you can ablate both.
        self.transform = nn.Linear(input_size, target_shape[-1])
        self.target_shape = target_shape

    def forward(self, x: torch.Tensor, mask=None) -> torch.Tensor:
        # TODO Qua dovrei fare pooling non reshaping.
        # todo redefine pooling strategy?
        # todo usa almeno masking per evitare di fare mean dove non va
        if len(x.shape) == 4 and (len(self.target_shape) == 3 or len(self.target_shape) == 2):
            x = x.mean(dim=-2)  # Avoid having parameters when matching KD head. (Student might cheat the teacher).

        if len(x.shape) == 3 and len(self.target_shape) == 2:
            x = x.mean(dim=-2)

        if len(x.shape) != len(self.target_shape):
            raise ValueError("Reshaping of tensor does not match target shape.")

        y = torch.nn.functional.normalize(self.transform(x), p=2, dim=-1)
        return y

from abc import ABC, abstractmethod

import torch
from torch import nn


class ModalityDropout(nn.Module, ABC):
    @abstractmethod
    def forward(self, b: int, device):
        pass


class DisabledModalityDropout(ModalityDropout):
    def __init__(self, supports_length: int, drop_p: float = 0):
        super(DisabledModalityDropout, self).__init__()
        self.supports_length: int = supports_length

    def forward(self, b: int, device):
        return torch.ones(b, self.supports_length, device=device)


class BernoulliSupportsModalityDropout(ModalityDropout):
    def __init__(self, supports_length: int, drop_p: float = 0):
        super(BernoulliSupportsModalityDropout, self).__init__()
        self.drop_p: float = drop_p
        self.supports_length: int = supports_length
        # Evaluate if exposing this
        self.ensure_one: bool = False

    def forward(self, b: int, device):
        if self.drop_p <= 0 or not self.training:
            # All valid so we return all
            return torch.ones(b, self.supports_length, device=device)

        # Bernoulli distribution to decide what to drop.
        keep = torch.bernoulli(torch.full((b, self.supports_length), 1 - self.drop_p, device=device)).bool()
        dead = ~keep.any(1)

        if self.ensure_one and dead:
            # If wanted we can restore at least one branch for training so that a supporting modalities is always available.
            summed_dead = int(dead.sum().item())
            keep[dead, torch.randint(0, self.supports_length, (summed_dead,), device=device)] = True

        return keep

import functools
from typing import Iterable, Callable

from torch import nn
from torchvision.transforms import Lambda

IDENTITY = Lambda(lambda x: x)


class Compose:
    def __init__(self, transforms: Iterable[nn.Module | Callable]):
        """
        Guide on how to write your own transforms: (torchvision v2 style).
        https://docs.pytorch.org/vision/main/auto_examples/transforms/plot_custom_transforms.html#sphx-glr-auto-examples-transforms-plot-custom-transforms-py

        :param transforms: Set of callable transforms in sequence
        """
        self.transforms: Iterable[nn.Module] = transforms

    def __call__(self, x):
        return functools.reduce(lambda d, t: t(d), self.transforms, x)

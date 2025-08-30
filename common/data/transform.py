import functools
from traceback import print_stack
from typing import Iterable, Callable

import torch
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

    def __call__(self, x, *args, **kwargs):
        return functools.reduce(lambda d, t: t(d, *args, **kwargs), self.transforms, x)

# todo ma è veramente utile? beh si dai
# TODO per random custom random augmentations non ho altro modo
class KwargsCompose(Compose):
    """
        Special compose that allows to update the kwargs running down the call stream.
        Can be used with normal calls without any problem. It supposes that the output is either:
        - A single x object (no metadata changed or persisted)
        - A tuple [x, metadata]. This updates the metadata object.
            If a transform has multiple outputs they have to be put according to the metadata limitation.
            ex. I return x and y -> (x,y) won't go! (x,y,metadata) also bad! -> ((x,y),metadata) is the correct formatting.
    """
    def __call__(self, x, *args, **kwargs):
        # Apply this seed to img transforms
        for t in self.transforms:
            try:
                x = t(x)
                if isinstance(x, tuple) and len(x) == 2:
                    x, upd_kwargs = x
                    kwargs = kwargs | upd_kwargs

            except Exception as e:
                print_stack()
                print("We had an exception when calling transform [", t.__class__.__name__, "]")

                raise e  # Propagate the exception further

        return x, kwargs

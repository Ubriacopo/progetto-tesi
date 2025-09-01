from abc import ABC, abstractmethod
from typing import Any

import torch
from torch import nn
from torchvision.transforms import Lambda

IDENTITY = Lambda(lambda x: x)


class FirstArgTransform(nn.Module, ABC):
    """
    For maximum compatibility
    https://docs.pytorch.org/vision/main/auto_examples/transforms/plot_custom_transforms.html

    Applies a transformation only to first element if the input is a tuple or to the element if it is not a tuple.
    """

    @classmethod
    def scriptable(cls) -> bool:
        return True

    def forward(self, x: Any):
        # If input is iterable we pass to call only first element (data)
        try:
            if type(x) == tuple or type(x) == list:
                mod = self.do(x[0])
                # If we output more objects they get added beyond the first
                if type(mod) == tuple or type(mod) == list:
                    return tuple([mod[0]] + list(x[1:]) + list(mod[1:]))

                x = list(x)
                x[0] = mod  # Update leftmost data stream
                return tuple(x)

            # X was just a value so we can call ourselves
            return self.do(x)

        except Exception as e:
            print("API level exception for transform forward")
            raise e

    @abstractmethod
    def do(self, x):
        pass

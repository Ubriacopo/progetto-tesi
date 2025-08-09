import functools
from abc import ABC
from typing import Any, Callable, List, Iterable

import torch


# https://github.com/KangHyunWook/Pytorch-implementation-of-Multimodal-emotion-recognition-on-RAVDESS-dataset
# https://docs.pytorch.org/tutorials/beginner/data_loading_tutorial.html?spm=a2c6h.13046898.publish-article.82.407b6ffabX8l7A
class Compose:
    # Like the one from torchvision.Compose([])
    def __init__(self,
                 pre: Iterable[Callable[[Any], Any]] = (),
                 aug: Iterable[Callable[[Any], Any]] = (),
                 post: Iterable[Callable[[Any], Any]] = ()):
        self.pre, self.aug, self.post = list(pre), list(aug), list(post)

    def __call__(self, x: Any, train: bool = True, return_both: bool = False) -> Any | tuple[Any, Any]:
        # After pre-processing was done we have pre
        # todo vedi se devi copiare
        pre = functools.reduce(lambda d, t: t(d), self.pre, x)
        y = pre

        if train and self.aug:
            # Augment
            pre_copy = pre.clone() if isinstance(pre, torch.Tensor) else pre
            y = functools.reduce(lambda d, t: t(d), self.aug, pre_copy)

        y = functools.reduce(lambda d, t: t(d), self.post, y)
        # Return pre and pre + optional(aug) + post. Only pre should be cached
        return (pre, y) if return_both else y

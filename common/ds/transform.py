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
        y = functools.reduce(lambda d, t: t(d), self.pre, x)
        return self.transform_but_skip_pre(y, train=train, return_both=return_both)

    def transform_but_skip_pre(self, x: Any, train: bool = True, return_both: bool = False) -> Any:
        y = x

        if train:
            # Augment only during training
            pre_copy = x.clone() if isinstance(x, torch.Tensor) else x
            y = functools.reduce(lambda d, t: t(d), self.aug, pre_copy)

        y = functools.reduce(lambda d, t: t(d), self.post, y)
        # Return pre and pre + optional(aug) + post. Only pre should be cached
        return (y, x) if return_both else y

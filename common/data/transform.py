import functools
from typing import Any, Callable, Iterable

import torch
from torchvision.transforms import Lambda

IDENTITY = Lambda(lambda x: x)


# todo drop pre- preprocessing is done elsewhere.
#       Scherzone lo tengo, magari voglio fare step prima di aug. ora tutta compose si limita a essere posprocessing  non salvabile.
#       Oppure piu pulito: faccio solo iterable e due compose distinte una train e una val. Avro in ogni caso due Dataset instances diversi.
class Compose:
    def __init__(
            self,
            pre: Iterable[Callable[[Any], Any]] = (),
            aug: Iterable[Callable[[Any], Any]] = (),
            post: Iterable[Callable[[Any], Any]] = ()
    ):
        """
        Like the one from torchvision.Compose([]).
        https://docs.pytorch.org/tutorials/beginner/data_loading_tutorial.html?spm=a2c6h.13046898.publish-article.82.407b6ffabX8l7A

        :param pre: Before augmentation steps.
        :param aug: Augmentation steps. Only if train=True.
        :param post: After augmentation steps.
        """
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
        # Return pre and pre + optional(aug) + post.
        # Only pre should be cached.
        return (y, x) if return_both else y

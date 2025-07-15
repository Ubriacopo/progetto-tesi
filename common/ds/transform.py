import functools
from abc import ABC
from typing import Any, Callable, List


# https://github.com/KangHyunWook/Pytorch-implementation-of-Multimodal-emotion-recognition-on-RAVDESS-dataset
# https://docs.pytorch.org/tutorials/beginner/data_loading_tutorial.html?spm=a2c6h.13046898.publish-article.82.407b6ffabX8l7A
class MultiModalTransform(ABC):
    def transform_video(self, video):
        pass


class Compose:
    # Like the one from torchvision.Compose([])
    def __init__(self, transforms: List[Callable[[Any], Any]]):
        self.transforms = transforms

    def __call__(self, data: Any) -> Any:
        return functools.reduce(lambda d, t: t(d), self.transforms, data)

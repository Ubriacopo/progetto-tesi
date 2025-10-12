import torch
from torch import nn

from main.core_data.media.assessment.assessment import Assessment


class RemapFieldToRange(nn.Module):
    def __init__(self, field_name: str, original_range: tuple[float, float], new_range: tuple[float, float] = (1, 9)):
        super().__init__()
        self.original_range: tuple[float, float] = original_range
        self.new_range: tuple[float, float] = new_range
        self.field_name: str = field_name

    def forward(self, x: Assessment) -> Assessment:
        if not hasattr(x, self.field_name):
            raise AttributeError("Field '{}' not found.".format(x))

        a, b = self.original_range
        c, d = self.new_range
        x.__setattr__(self.field_name, (getattr(x, self.field_name) - a) * d / b)
        return x


class SliceAssessments(nn.Module):
    def __init__(self, max_idx: int):
        super().__init__()
        self.max_idx: int = max_idx

    def forward(self, x: Assessment) -> Assessment:
        x.data = x.data[:self.max_idx]
        return x


class ToTensorData(nn.Module):
    # noinspection PyMethodMayBeStatic
    def forward(self, x: Assessment):
        x.data = torch.tensor(x.data)
        return x


class PermuteAssessments(nn.Module):
    def __init__(self, original_order: str, target_order: str = Assessment.default_order()):
        super().__init__()
        self.original_order: str = original_order.replace(" ", "")
        self.target_order: str = target_order.replace(" ", "")

    def forward(self, x: Assessment):
        indices = torch.tensor([self.original_order.index(item) for item in self.target_order])
        x.data = x.data[indices]
        return x

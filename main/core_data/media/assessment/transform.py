from dataclasses import asdict

from torch import nn

from main.core_data.media.assessment.assessment import Assessment


class ToObjectDict(nn.Module):
    # noinspection PyMethodMayBeStatic
    def forward(self, x: Assessment):
        return asdict(x)


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

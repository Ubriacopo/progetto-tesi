from typing import Tuple

import numpy as np
from tensordict import TensorDict
from torch import nn

from main.core_data.media.assessment.assessment import Assessment, AssessmentLabels


class AssessmentToTensor(nn.Module):
    # noinspection PyMethodMayBeStatic
    def forward(self, x: Assessment) -> TensorDict:
        x.data = {"scores": x.data, "labels": x.labels, "scales": x.scales}
        return TensorDict(x.data)


class RescaleAssessmentValue(nn.Module):
    def __init__(self, key: str, rescale_range: Tuple[float | int, float | int] = (0, 1)):
        super().__init__()
        self.key: str = key
        self.rescale_range: Tuple[float | int, float | int] = rescale_range

    def forward(self, x: Assessment) -> Assessment:
        if self.key not in x.labels:
            return x

        idx = x.labels.index(self.key)
        a, b = x.scales[idx]

        if b == a:
            raise ValueError(f"Invalid source scale: {(a, b)}")

        x.scales[idx] = self.rescale_range
        c, d = self.rescale_range

        start_x = x.data[idx]
        new_x = c + ((start_x - a) * (d - c)) / (b - a)
        if type(c) is int and type(d) is int:
            x.data[idx] = np.rint(new_x).astype(x.data.dtype)

        return x

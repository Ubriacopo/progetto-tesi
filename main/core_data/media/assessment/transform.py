from tensordict import TensorDict
from torch import nn

from main.core_data.media.assessment.assessment import Assessment


class AssessmentToTensor(nn.Module):
    # noinspection PyMethodMayBeStatic
    def forward(self, x: Assessment) -> TensorDict:
        x.data = {"scores": x.data, "labels": x.labels, "scales": x.scales}
        return TensorDict(x.data)

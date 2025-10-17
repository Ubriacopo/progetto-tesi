from tensordict import TensorDict
from torch import nn

from main.core_data.media.metadata.metadata import Metadata


class MetadataToTensor(nn.Module):
    # noinspection PyMethodMayBeStatic
    def forward(self, x: Metadata):
        if x.interval is not None:
            x.data["interval"] = x.interval
        return TensorDict(x.data)

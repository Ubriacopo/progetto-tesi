# TODO: During runtime (loading) pad to batch max length
#   Then you have to only pass valid steps to the backbone via masking
import torch
from tensordict import TensorDict
from torch import nn

from main.model.neegavi.model import EegInterAviModel

# todo rename to simply linear probe and move?
class FusionLinearProbe(nn.Module):
    def __init__(self, backbone: EegInterAviModel, in_dim: int, out_dim: int):
        super(FusionLinearProbe, self).__init__()
        self.backbone: EegInterAviModel = backbone
        self.head: nn.Module = nn.Linear(in_dim, out_dim)

    def forward(self, x: TensorDict) -> torch.Tensor:
        b_inner = x.shape[1]
        x = x.flatten(0, 1)
        with torch.no_grad():
            y = self.backbone(x)  # TODO eeginteravi inference only model not all
        # Restore batch
        y = y.cls.unflatten(0, (-1, b_inner))
        # Pool
        y = y.mean(dim=-2)
        return self.head(y)

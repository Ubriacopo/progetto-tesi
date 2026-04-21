from einops.layers.torch import Rearrange
from torch import nn

from main.model.downstream.core.model.finetune import EegAviFineTune
from main.model.neegavi.model import EegInterAviModel


class DefaultEavFineTune(EegAviFineTune):
    def __init__(self, encoder: EegInterAviModel, num_classes: int = 5):
        self.classes = num_classes
        super().__init__(encoder)

    def make_projection_head(self) -> nn.Module:
        return nn.Sequential(
            Rearrange('b t d -> b (t d)'),
            nn.Linear(2 * 384, 200),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(200, self.classes),
        )

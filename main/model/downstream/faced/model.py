from cbramod.models.cbramod import CBraMod
from einops.layers.torch import Rearrange
from torch import nn

from main.model.downstream.core.model.finetune import CBraFineTune


class DefaultFacedCBraFineTune(CBraFineTune):
    def __init__(self, encoder: CBraMod, num_classes: int = 9):
        super().__init__(encoder)
        self.classes = num_classes

    def make_projection_head(self) -> nn.Module:
        return nn.Sequential(
            Rearrange('b c s d -> b (c s d)'),
            nn.Linear(32 * 10 * 200, 10 * 200),
            nn.ELU(),
            nn.Dropout(0.2),
            nn.Linear(10 * 200, 200),
            nn.ELU(),
            nn.Dropout(0.2),
            nn.Linear(200, self.classes),
        )

from cbramod.models.cbramod import CBraMod
from einops.layers.torch import Rearrange, Reduce
from torch import nn

from main.model.downstream.core.model.finetune import CBraFineTune, EegAviFineTune
from main.model.downstream.core.model.linear_probe import EegAviProbe, CBraModProbe
from main.model.neegavi.model import EegInterAviModel


class DefaultFacedCBraFineTune(CBraFineTune):
    def __init__(self, encoder: CBraMod, num_classes: int = 9):
        self.classes = num_classes
        super().__init__(encoder)

    def make_projection_head(self) -> nn.Module:
        return nn.Sequential(
            Reduce("B b c s d -> B c s d", "max"),
            Rearrange('b c s d -> b (c s d)'),
            nn.Linear(32 * 19 * 200, 200),
            nn.ELU(),
            nn.Dropout(0.2),
            nn.Linear(200, self.classes),
        )


class DefaultFacedCBraModLinearProbe(CBraModProbe):
    def __init__(self, encoder: CBraMod, num_classes: int = 9):
        self.classes = num_classes
        super().__init__(encoder)

    def make_projection_head(self) -> nn.Module:
        return nn.Sequential(
            Reduce("B b c s d -> B c s d", "max"),
            Rearrange('b c s d -> b (c s d)'),
            nn.Linear(32 * 19 * 200, self.classes),
        )


class DefaultFacedFineTune(EegAviFineTune):
    def __init__(self, encoder: EegInterAviModel, num_classes: int = 9):
        self.classes = num_classes
        super().__init__(encoder)

    def make_projection_head(self) -> nn.Module:
        return nn.Sequential(
            Rearrange('b t d -> b (t d)'),
            nn.Linear(3 * 384, 200),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(200, self.classes),
        )


class DefaultFacedLinearProbe(EegAviProbe):
    def __init__(self, encoder: EegInterAviModel, num_classes: int = 9):
        self.classes = num_classes
        super().__init__(encoder)

    def make_projection_head(self) -> nn.Module:
        return nn.Sequential(
            Rearrange('b t d -> b (t d)'),
            nn.Linear(3 * 384, self.classes),
        )

from abc import ABC

from cbramod.models.cbramod import CBraMod

from main.model.downstream.core.model.base import EegAviBaseModel, CBraModBaseModel
from main.model.neegavi.model import EegInterAviModel


class EegAviProbe(EegAviBaseModel, ABC):
    def __init__(self, encoder: EegInterAviModel):
        super().__init__(encoder, True)

    def freeze(self):
        for p in self.encoder.parameters():
            p.requires_grad = False


class CBraModProbe(CBraModBaseModel, ABC):
    def __init__(self, encoder: CBraMod):
        super().__init__(encoder, True)

    def freeze(self):
        for p in self.encoder.parameters():
            p.requires_grad = False

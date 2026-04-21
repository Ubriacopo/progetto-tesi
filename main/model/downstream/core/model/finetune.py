from abc import ABC

from cbramod.models.cbramod import CBraMod
from tensordict import TensorDict
from torch import nn

from main.model.downstream.core.model.base import BaseModel, EegAviBaseModel, CBraModBaseModel
from main.model.neegavi.adapters import EegCbraModAdapter
from main.model.neegavi.model import EegInterAviModel


class EegAviFineTune(EegAviBaseModel, ABC):
    def __init__(self, encoder: EegInterAviModel):
        super().__init__(encoder, False)

    def get_pivot_adapter(self) -> EegCbraModAdapter:
        if not isinstance(self.encoder.pivot.adapter, EegCbraModAdapter):
            raise NotImplementedError("Can only accept cbra backbone")
        return self.encoder.pivot.adapter

    def freeze(self):
        cbra: EegCbraModAdapter = self.get_pivot_adapter()
        for p in cbra.encoder.encoder.parameters():
            p.requires_grad = False

        # Unfreeze what we want to learn
        for l in cbra.encoder.encoder.layers[-2:]:
            for p in l.parameters():
                p.requires_grad = True


class CBraFineTune(CBraModBaseModel, ABC):
    def __init__(self, encoder: CBraMod):
        super().__init__(encoder, False)

    def freeze(self):
        # Freeze everything
        # for p in self.encoder.parameters():
        #    p.requires_grad = False

        # Unfreeze what we want to learn
        for l in self.encoder.encoder.layers[-2:]:
            for p in l.parameters():
                p.requires_grad = True
        return

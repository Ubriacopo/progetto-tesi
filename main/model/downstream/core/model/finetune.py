from abc import abstractmethod, ABC

from cbramod.models.cbramod import CBraMod
from tensordict import TensorDict
from torch import nn

from main.model.neegavi.adapters import EegCbraModAdapter
from main.model.neegavi.model import EegInterAviModel


class FineTuneModel(nn.Module, ABC):
    @abstractmethod
    def make_projection_head(self) -> nn.Module:
        pass

    @abstractmethod
    def freeze(self):
        pass


class EegAviFineTune(FineTuneModel, ABC):
    def __init__(self, encoder: EegInterAviModel):
        super().__init__()
        self.encoder: EegInterAviModel = encoder
        if not isinstance(self.encoder.pivot.adapter, EegCbraModAdapter):
            raise NotImplementedError("Can only accept cbra backbone")
        self.encoder.pivot.adapter.encoder.proj_out = nn.Identity()
        self.project = self.make_projection_head()

    def freeze(self):
        if not isinstance(self.encoder.pivot.adapter, EegCbraModAdapter):
            raise NotImplementedError("Can only accept cbra backbone")

        cbra: EegCbraModAdapter = self.encoder.pivot.adapter

        for p in cbra.encoder.encoder.parameters():
            p.requires_grad = False

        # Unfreeze what we want to learn
        for l in cbra.encoder.encoder.layers[-2:]:
            for p in l.parameters():
                p.requires_grad = True

    def forward(self, x: TensorDict):
        b, b_inner = x.shape[:2]
        # Collapse the batch size
        x = x.flatten(0, 1)
        y = self.encoder(x)
        # Restore the batch size and use the CLS token
        z = y.cls.unflatten(0, (b, b_inner))
        logits = self.project(z)
        return logits


class CBraFineTune(FineTuneModel, ABC):
    def __init__(self, encoder: CBraMod):
        super().__init__()
        self.encoder: CBraMod = encoder
        self.encoder.proj_out = nn.Identity()
        self.project = self.make_projection_head()

        self.freeze()

    def freeze(self):
        # Freeze everything
        for p in self.encoder.parameters():
            p.requires_grad = False

        # Unfreeze what we want to learn
        for l in self.encoder.encoder.layers[-2:]:
            for p in l.parameters():
                p.requires_grad = True

    @abstractmethod
    def forward(self, td: TensorDict):
        x, mask = td["eeg", "data"], td["eeg", "mask"]
        b, b_inner = x.shape[:2]
        # Collapse the batch
        x = x.flatten(0, 1)
        mask_flat = mask.flatten(0, 1).bool()
        y = self.encoder(x, mask)
        z = y.cls.unflatten(0, (b, b_inner))
        # TODO mask?
        logits = self.project(z)
        return logits

from abc import ABC, abstractmethod

import torch
from cbramod.models.cbramod import CBraMod
from tensordict import TensorDict
from torch import nn

from main.model.neegavi.adapters import EegCbraModAdapter
from main.model.neegavi.model import EegInterAviModel


class BaseModel(nn.Module, ABC):
    @abstractmethod
    def make_projection_head(self) -> nn.Module:
        pass

    @abstractmethod
    def freeze(self):
        pass


class EegAviBaseModel(BaseModel, ABC):
    def __init__(self, encoder: EegInterAviModel, encoder_frozen: bool):
        super().__init__()
        self.encoder: EegInterAviModel = encoder
        if not isinstance(self.encoder.pivot.adapter, EegCbraModAdapter):
            raise NotImplementedError("Can only accept cbra backbone")
        self.encoder.pivot.adapter.encoder.proj_out = nn.Identity()
        self.project = self.make_projection_head()
        self.encoder_frozen: bool = encoder_frozen
        # Freeze parts of the model you want to keep fixed
        self.freeze()

        if self.encoder_frozen:
            self.encoder.eval()

    def get_pivot_adapter(self) -> EegCbraModAdapter:
        if not isinstance(self.encoder.pivot.adapter, EegCbraModAdapter):
            raise NotImplementedError("Can only accept cbra backbone")
        return self.encoder.pivot.adapter

    def forward(self, x: TensorDict):
        b, b_inner = x.shape[:2]
        # Collapse the batch size
        x = x.flatten(0, 1)

        if self.encoder_frozen:
            with torch.no_grad():
                y = self.encoder(x)
        else:
            y = self.encoder(x)

        # Restore the batch size and use the CLS token
        z = y.cls.unflatten(0, (b, b_inner))
        logits = self.project(z)
        return logits


class CBraModBaseModel(BaseModel, ABC):
    def __init__(self, encoder: CBraMod, encoder_frozen: bool):
        super().__init__()
        self.encoder: CBraMod = encoder
        self.encoder.proj_out = nn.Identity()
        self.project = self.make_projection_head()
        self.encoder_frozen: bool = encoder_frozen
        # Freeze parts of the model you want to keep fixed
        self.freeze()

        if self.encoder_frozen:
            self.encoder.eval()

    def forward(self, td: TensorDict):
        x, mask = td["eeg", "data"], td["eeg", "mask"]
        b, b_inner = x.shape[:2]
        # Collapse the batch
        x = x.flatten(0, 1)
        mask_flat = ~mask.flatten(0, 1).bool()

        if self.encoder_frozen:
            with torch.no_grad():
                y = self.encoder(x.float(), mask_flat)
        else:
            y = self.encoder(x.float(), mask_flat)

        z = y.unflatten(0, (b, b_inner))
        # valid = ~mask.bool()
        # z = z * valid.unsqueeze(-1)
        logits = self.project(z.half())
        return logits

import torch
from cbramod.models.cbramod import CBraMod
from einops.layers.torch import Reduce, Rearrange
from tensordict import TensorDict
from torch import nn

from main.model.downstream.core.model.finetune import EegAviFineTune, CBraFineTune
from main.model.downstream.core.model.linear_probe import CBraModProbe, EegAviProbe
from main.model.neegavi.model import EegInterAviModel


class DefaultClareCBraFineTune(CBraFineTune):
    def __init__(self, encoder: CBraMod, num_classes: int = 2):
        self.classes = num_classes
        super().__init__(encoder)

    def make_projection_head(self) -> nn.Module:
        return nn.Sequential(
            Rearrange('b c s d -> b (c s d)'),
            nn.Linear(32 * 19 * 200, 200),
            nn.ELU(),
            nn.Dropout(0.2),
            nn.Linear(200, self.classes),
        )

    def forward(self, td: TensorDict):
        x, mask = td["eeg", "data"], td["eeg", "mask"]
        # Collapse the batch
        mask_flat = ~mask.bool()

        if self.encoder_frozen:
            with torch.no_grad():
                y = self.encoder(x.float(), mask_flat)
        else:
            y = self.encoder(x.float(), mask_flat)

        # valid = ~mask.bool()
        # z = z * valid.unsqueeze(-1)
        y = torch.nan_to_num(y, nan=0.0)
        logits = self.project(y.half())
        return logits


class DefaultClareBraModLinearProbe(CBraModProbe):
    def __init__(self, encoder: CBraMod, num_classes: int = 2):
        self.classes = num_classes
        super().__init__(encoder)

    def make_projection_head(self) -> nn.Module:
        return nn.Sequential(
            Rearrange('b c s d -> b (c s d)'),
            nn.Linear(32 * 19 * 200, self.classes),
        )

    def forward(self, td: TensorDict):
        x, mask = td["eeg", "data"], td["eeg", "mask"]
        # Collapse the batch
        mask_flat = ~mask.bool()

        if self.encoder_frozen:
            with torch.no_grad():
                y = self.encoder(x.float(), mask_flat)
        else:
            y = self.encoder(x.float(), mask_flat)

        # valid = ~mask.bool()
        # z = z * valid.unsqueeze(-1)
        y = torch.nan_to_num(y, nan=0.0)
        logits = self.project(y.half())
        return logits


class DefaultClareFineTune(EegAviFineTune):
    def __init__(self, encoder: EegInterAviModel, num_classes: int = 2):
        self.classes = num_classes
        super().__init__(encoder)

    def make_projection_head(self) -> nn.Module:
        return nn.Sequential(
            nn.Linear(384, 200),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(200, self.classes),
        )

    def forward(self, x: TensorDict):
        if self.encoder_frozen:
            with torch.no_grad():
                y = self.encoder(x)
        else:
            y = self.encoder(x)
        # Restore the batch size and use the CLS token
        z = y.cls
        z = torch.nan_to_num(z, nan=0.0)
        logits = self.project(z)
        return logits


class DefaultClareLinearProbe(EegAviProbe):
    def __init__(self, encoder: EegInterAviModel, num_classes: int = 2):
        self.classes = num_classes
        super().__init__(encoder)

    def make_projection_head(self) -> nn.Module:
        return nn.Sequential(
            nn.Linear(384, self.classes),
        )

    def forward(self, x: TensorDict):
        if self.encoder_frozen:
            with torch.no_grad():
                y = self.encoder(x)
        else:
            y = self.encoder(x)
        # Restore the batch size and use the CLS token
        z = y.cls
        z = torch.nan_to_num(z, nan=0.0)
        logits = self.project(z)
        return logits

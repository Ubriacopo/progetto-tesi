import torch
from cbramod.models.cbramod import CBraMod
from einops import rearrange
from torch import nn

from common.model.embedding.foundation_embedder import FoundationEmbedder
from common.model.utils import freeze_module


class CBraModFoundationEmbedder(FoundationEmbedder):
    def __init__(self, output_size: int = 200, device=None, freeze: bool = True,
                 weights: str = "../../../dependencies/cbramod/pretrained_weights.pth"):
        model = CBraMod()
        if device is None:
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        assert weights is not None or freeze is False, "If you freeze the model has to have a state"
        if weights is not None:
            model.load_state_dict(torch.load(weights, map_location=device))

        super().__init__(model, output_size, freeze)

    def forward(self, x, mask=None) -> torch.Tensor:
        if self.model_is_frozen:
            with torch.no_grad():
                x = self.base_model(x.float())
        else:
            x = self.base_model(x.float())

        return x

    def get_output_shape(self) -> tuple[int, ...]:
        return -1, 1, 1, 1  # todo in futuro pulire qui


# todo fuse in one
class CBraModFoundationEmbedderForTimeSequences(nn.Module):
    def __init__(self, output_size: int = 200, device=None, freeze: bool = True,
                 weights: str = "../../dependencies/cbramod/pretrained_weights.pth"):
        super().__init__()
        model = CBraMod()
        if device is None:
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        assert weights is not None or freeze is False, "If you freeze the model has to have a state"
        if weights is not None:
            model.load_state_dict(torch.load(weights, map_location=device))

        self.model_is_frozen: bool = freeze
        if freeze:
            freeze_module(model)

        self.base_model = model
        self.output_size: int = output_size

    # TODO MASKING
    def forward(self, x, mask=None) -> torch.Tensor:
        b = x.shape[0]  # Batch size
        y = rearrange(x, "b T f p d -> (b T) f p d")
        y = y.float()

        if self.model_is_frozen:
            with torch.no_grad():
                y = self.base_model(y)
        else:
            y = self.base_model(y)

        y = rearrange(y, "(b T) f F D -> b T f F D", b=b)
        return y

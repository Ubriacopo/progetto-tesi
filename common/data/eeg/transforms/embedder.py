import torch
from cbramod.models.cbramod import CBraMod
from torch import nn


class CBraModEmbedderTransform(nn.Module):
    def __init__(self, weights_path: str = "../../../dependencies/cbramod/pretrained_weights.pth",
                 device=None, **kwargs):
        super().__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") if device is None else device
        self.model = CBraMod(**kwargs).to(device)

        if weights_path is not None:
            self.model.load_state_dict(torch.load(weights_path, map_location=self.device))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO: Vedi se gestisce batch.
        with torch.inference_mode():
            y = self.model(x.float().to(self.device))
        return y

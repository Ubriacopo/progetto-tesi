import torch
from cbramod.models.cbramod import CBraMod

from common.model.embedding.foundation_embedder import FoundationEmbedder


class CBraModFoundationEmbedder(FoundationEmbedder):

    def __init__(self, output_size: int = 200, device=None, freeze: bool = True,
                 weights: str = "../../dependencies/cbramod/pretrained_weights.pth"):
        model = CBraMod()
        if device is None:
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        assert weights is not None or freeze is False, "If you freeze the model has to have a state"
        if weights is not None:
            model.load_state_dict(torch.load(weights, map_location=device))

        super().__init__(model, output_size, freeze)

    def retrieve_patches(self, x):
        return x

    def reshape_for_perceiver(self, x):
        return x

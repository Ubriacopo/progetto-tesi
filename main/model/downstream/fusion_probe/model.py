from torch import nn

from main.model.downstream.linear_probe import SimpleLinearProbe
from main.model.neegavi.model import EegInterAviModel

# AGGIUNGI DREAMER x avere piu sample
class ValArousal5FusionLinearProbe(SimpleLinearProbe):
    def __init__(self, backbone: EegInterAviModel, in_dim: int = 384, out_dim: int = 5):
        super(ValArousal5FusionLinearProbe, self).__init__(backbone, in_dim, out_dim)
        self.backbone: EegInterAviModel = backbone
        self.head: nn.Module = nn.Linear(in_dim, out_dim)

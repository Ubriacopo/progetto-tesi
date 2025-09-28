from torch import nn


# For a lightweight ablation or if you just want to test gating as a concept:
# Use the simple global sigmoid scalar per modality — perfectly fine to start with.
class GateUnit(nn.Module):
    def __init__(self, supporting_size_embedding, modality_mappings):
        super(GateUnit, self).__init__()


class DeepGateUnit(nn.Module):
    def __init__(self, supporting_size_embedding, modality_mappings):
        super(DeepGateUnit, self).__init__()

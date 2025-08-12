import torch


# For first approach we try using a pre-defined embedding model.
class FEEG(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.layers = []
        # Structure it

    def forward(self, x):
        eeg, video, audio, text = x
        # Put the embeders in the model. Reasonable
        # If I see it takes too long each iteration just pre-processs the data elsewhere
        # Efficacia impresecendibile Efficenza opinabile -> Efficenza dopo

        # For video we use ViViT kinetics
        # For audio we use Hubert
        # For text
        # For EEG none (We distill from a model)

        return 0

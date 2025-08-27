import torch


class ToMono:
    """
    Transforms a source from Stereo or any other format to MONO. (Single wave)
    """

    def __call__(self, x: torch.Tensor, **kwargs):
        assert isinstance(x, torch.Tensor), "Input must be a torch.Tensor"
        return torch.mean(x, dim=0, keepdim=True)

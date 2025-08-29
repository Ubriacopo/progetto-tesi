import torch


class ToMono:
    """
    Transforms a source from Stereo or any other format to MONO. (Single wave)
    """

    def __call__(self, x: torch.Tensor, **kwargs):
        if not isinstance(x, torch.Tensor):
            raise TypeError("Expected a torch.Tensor")

        return torch.mean(x, dim=0, keepdim=True)

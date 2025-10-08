import torch
from torch import Tensor
from torch.nn.functional import normalize, logsigmoid, binary_cross_entropy_with_logits


def siglip(za: Tensor, zb: Tensor, logt: Tensor = torch.log(Tensor([10])), bias: Tensor = Tensor([-10])):
    """
    We will use this one.
    Implemented from Algorithm 1 Sigmoid loss pseudo-implementation by
    https://arxiv.org/pdf/2303.15343

    And seen in the official google repository:
    https://github.com/google-research/big_vision/blob/main/big_vision/trainers/proj/image_text/siglip.py

    :param za: One modality's embeddings.
    :param zb: Other modality's embeddings.
    :param logt: Temperature initialization. (Is the exponent of e^temperature).
            It should be a learnable parameter as per SigLIP paper. Defaults to log(10)
    :param bias: Bias initialization. It should be a learnable parameter as per SigLIP paper.
            Defaults to -10.
    :return: The calculated Sigmoid loss. (Its name comes from the original model it was used for).
            We adapted (or are working on it to work) for EEG data mixed with video, text and audio.
    """
    T = torch.exp(logt).to(za.device)
    B = za.shape[0]

    # L2-Normalization
    za = normalize(za, p=2, dim=-1)
    zb = normalize(zb, p=2, dim=-1)

    logits = (za @ zb.T) * T + bias.to(za.device)
    labels = 2 * torch.eye(B, device=za.device) - torch.ones(B, device=za.device)
    loss = -torch.sum(logsigmoid(logits * labels), dim=-1).mean()
    return loss

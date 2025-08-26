import torch
from torch import Tensor
from torch.nn.functional import normalize, logsigmoid, binary_cross_entropy_with_logits


def sigLIP(za: Tensor, zb: Tensor, logt: Tensor = torch.log(Tensor([10])), bias: Tensor = Tensor([-10])):
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
    T = torch.exp(logt)
    B = za.shape[0]

    # L2-Normalization
    za = normalize(za, p=2, dim=-1)
    zb = normalize(zb, p=2, dim=-1)

    logits = (za @ zb.T) * T + bias
    labels = 2 * torch.eye(B) - torch.ones(B)
    loss = -torch.sum(logsigmoid(logits * labels), dim=-1).mean()
    return loss


def sig_LIP_logits(emb_a: Tensor, emb_b: Tensor, t_prime: Tensor, bias, reduction: str = None):
    """
    Retouched version of SigLIP that works with the torch builtin BCE.
    To reproduce SigLIP original output one has to pass as reduction="sum"

    :param emb_a:
    :param emb_b:
    :param t_prime:
    :param bias:
    :param reduction:
    :return:
    """
    T = torch.exp(t_prime)

    if reduction is None:
        reduction = "mean"

    # L2-normalization
    z_a = normalize(emb_a, p=2, dim=-1)
    z_b = normalize(emb_b, p=2, dim=-1)

    logits = (z_a @ z_b.T) * T + bias

    pos = logits.diag()
    positive_loss = binary_cross_entropy_with_logits(pos, torch.ones_like(pos), reduction=reduction)

    neg = logits[~torch.eye(logits.size(0), dtype=torch.bool, device=logits.device)]
    negative_loss = binary_cross_entropy_with_logits(neg, torch.zeros_like(neg), reduction=reduction)
    return positive_loss + negative_loss

import math
from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn.functional import normalize, logsigmoid

from main.utils.logging import make_logger


@torch.no_grad()
def siglip_random_baseline(loss_fn, a, b):
    # shuffle targets to break alignment
    idx = torch.randperm(b.shape[0], device=b.device)
    return loss_fn(a, b[idx])


class SiglipLoss(nn.Module):
    def __init__(
            self,
            init_tau=0.07,
            init_bias=-10,
            tau_min: float = 0.01,
            tau_max: float = 0.5,
            bias_scale: float = 5.0,
            stop_grad_target: bool = False,
    ):
        super(SiglipLoss, self).__init__()
        self.stop_grad_target: bool = stop_grad_target
        self.logger = make_logger(self.__class__.__name__)
        if self.stop_grad_target:
            self.logger.info("Heads will be detached for forward pass in this class instance")

        self.LOGT_MIN = math.log(1 / tau_max)
        self.LOGT_MAX = math.log(1 / tau_min)

        self.bias_scale = float(bias_scale)

        init_logt = math.log(1.0 / float(init_tau))
        p = (init_logt - self.LOGT_MIN) / (self.LOGT_MAX - self.LOGT_MIN)
        p = min(max(p, 1e-4), 1.0 - 1e-4)  # keep invertible
        init_u = math.log(p / (1.0 - p))
        self.logt = nn.Parameter(torch.tensor([init_u], dtype=torch.float32))
        self.bias = nn.Parameter(torch.tensor([init_bias], dtype=torch.float32))

    def forward(self, za: torch.Tensor, zb: torch.Tensor, zb_negative: Optional[torch.Tensor] = None, ignore_mask=None):
        """
        How does siglip work:

        L = - 1/N sum(log(sim(x_ii / tau)) + b + 1 / (N+1) sum(log(1 - sim(x_ij / tau + b))
        where sim is log-sigmoid = -log(1 - e^-x)

        Bias controls is an offset value on the input of the sigmoid changing where the loss operates.
        Larger bias logits are more negative -> Reduces flatness (In some correct ranges)
        Generally speaking the bias helps to control the region of negatives to avoid saturating gradients too soon.

        tau temperature instead rescales similarity before the sigmoid or softmax.
        Higher taus mean more stable training but also weaker gradients with smoother probs (logits are smaller).
        Lower taus mean larger logits and easier to overfit or saturate (hit saturation earlier is a big problem).

        Compared to InfoNCE has fewer issues with smaller batches.

        :param zb_negative:
        :param za:
        :param zb:
        :param ignore_mask:
        :return:
        """
        # Normalization
        za = F.normalize(za, dim=-1)
        if self.stop_grad_target:
            zb = zb.detach()

        zb = F.normalize(zb, dim=-1)
        if zb_negative is not None:
            zb_negative = F.normalize(zb_negative, dim=-1)
            zb_negative = zb_negative.detach()
            zb = torch.cat([zb, zb_negative], dim=0)

        t = (self.LOGT_MIN + (self.LOGT_MAX - self.LOGT_MIN) * torch.sigmoid(self.logt)).exp()
        bias = self.bias_scale * torch.tanh(self.bias / self.bias_scale)

        logits = (za @ zb.T) * t + bias  # [B, B]

        B = za.size(0)
        M = logits.size(1)
        # +1 on diag, -1 off-diag

        labels = -torch.ones((B, M), device=logits.device, dtype=logits.dtype)
        labels[torch.arange(B, device=logits.device), torch.arange(B, device=logits.device)] = 1.
        loss_mat = -logsigmoid(logits * labels)  # [B, M]
        if ignore_mask is not None:
            assert ignore_mask.shape == (B, B) and ignore_mask.dtype == torch.bool
            loss_mat[:, :B] = loss_mat[:, :B].masked_fill(ignore_mask, 0.0)

            valid_per_query = M - ignore_mask.sum(dim=1)  # [B]
            per_query_loss = loss_mat.sum(dim=1) / valid_per_query.clamp_min(1)

            return per_query_loss.mean()

        return loss_mat.sum(dim=-1).mean()


class InfoNCE(nn.Module):
    """
    Calculates the InfoNCE loss for self-supervised learning.
    This contrastive loss enforces the embeddings of similar (positive) samples to be close
        and those of different (negative) samples to be distant.
    A query embedding is compared with one positive key and with one or more negative keys.

    References:
        https://arxiv.org/abs/1807.03748v2
        https://arxiv.org/abs/2010.05113

    Args:
        temperature: Logits are divided by temperature before calculating the cross entropy.
        reduction: Reduction method applied to the output.
            Value must be one of ['none', 'sum', 'mean'].
            See torch.nn.functional.cross_entropy for more details about each option.
        negative_mode: Determines how the (optional) negative_keys are handled.
            Value must be one of ['paired', 'unpaired'].
            If 'paired', then each query sample is paired with a number of negative keys.
            Comparable to a triplet loss, but with multiple negatives per sample.
            If 'unpaired', then the set of negative keys are all unrelated to any positive key.

    Input shape:
        query: (N, D) Tensor with query samples (e.g. embeddings of the input).
        positive_key: (N, D) Tensor with positive samples (e.g. embeddings of augmented input).
        negative_keys (optional): Tensor with negative samples (e.g. embeddings of other inputs)
            If negative_mode = 'paired', then negative_keys is a (N, M, D) Tensor.
            If negative_mode = 'unpaired', then negative_keys is a (M, D) Tensor.
            If None, then the negative keys for a sample are the positive keys for the other samples.

    Returns:
         Value of the InfoNCE Loss.

     Examples:
        >>> loss = InfoNCE()
        >>> batch_size, num_negative, embedding_size = 32, 48, 128
        >>> query = torch.randn(batch_size, embedding_size)
        >>> positive_key = torch.randn(batch_size, embedding_size)
        >>> negative_keys = torch.randn(num_negative, embedding_size)
        >>> output = loss(query, positive_key, negative_keys)
    """

    def __init__(self, temperature=0.05, reduction='mean', negative_mode='unpaired'):
        super().__init__()
        self.temperature = temperature
        self.reduction = reduction
        self.negative_mode = negative_mode

    def forward(self, query, positive_key, negative_keys=None):
        return info_nce(query, positive_key, negative_keys, temperature=self.temperature,
                        reduction=self.reduction, negative_mode=self.negative_mode)


def transpose(x):
    return x.transpose(-2, -1)


def masked_cosine_similarity(za: Tensor, zb: Tensor, present: Tensor):
    w = present.float()
    # Masked mean of (1 - cos)
    return ((1 - F.cosine_similarity(za, zb, dim=-1)) * w).sum() / w.sum().clamp_min(1.0)

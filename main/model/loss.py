import torch
from torch import Tensor, nn
from torch.nn.functional import normalize, logsigmoid, binary_cross_entropy_with_logits
import torch.nn.functional as F


def lightweight_whitening(z: torch.Tensor):
    """
    common component removal / mean-centering (lightweight version of whitening that works great for small-batch contrastive training).
    Remember: this makes the loss batch-dependent (the center is computed over the current batch).
    Usually fine in practice. (no good for multi GPUs)

    As seen in SimCLR / MoCo-v3 / BYOL-A and formulated in Whitening Contrastive Learning (WCL, CVPR 2021)
    TODO: Read paper

    :param z: Tensor to normalize
    :return: The normalized z centered on mean
    """
    # z = F.normalize(z, dim=-1)
    z = F.normalize(z - z.mean(0, keepdim=True), dim=-1)  # optional but stabilizes small-batch
    return z


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
    labels = 2 * torch.eye(B, device=za.device) - 1

    # This is original loss computation from siglip proposal
    loss = -torch.sum(logsigmoid(logits * labels), dim=-1).mean()
    # But since we work with multi-losses we have to scale down to factor B^2 not B
    # loss = -torch.mean(logsigmoid(logits * labels))
    return loss


class SiglipLoss(nn.Module):
    def __init__(self, init_tau=0.07, init_bias=-10, stop_grad_target: bool = False, verbose: bool = False):
        super(SiglipLoss, self).__init__()

        self.logt = nn.Parameter(torch.tensor([float(torch.log(torch.tensor(1.0 / init_tau)))]))  # ~ ln(1/τ)
        # self.logt = torch.log(torch.tensor([1. / init_tau], device="cuda"))
        # learnable scalar bias (start near 0 so positives can go > 0)
        self.bias = nn.Parameter(torch.tensor([init_bias], dtype=torch.float32))
        # self.bias = torch.tensor([init_bias], device="cuda")
        self.verbose = verbose
        self.stop_grad_target: bool = stop_grad_target

    def forward(self, za: torch.Tensor, zb: torch.Tensor, ignore_mask=None):
        # Normalization
        za = lightweight_whitening(za)
        if self.stop_grad_target:
            self.verbose and print("Head has been detached")
            zb = zb.detach()

        zb = lightweight_whitening(zb)

        b = self.bias
        T = self.logt.exp()
        logits = (za @ zb.T) * T + b  # [B, B]
        B = logits.size(0)
        # +1 on diag, -1 off-diag
        labels = 2 * torch.eye(B, device=logits.device, dtype=logits.dtype) - 1  # [+1 diag, -1 off]
        if ignore_mask is not None:
            # ignore_mask: True where we want to drop loss (e.g., duplicates off-diag)
            logits = logits.masked_fill(ignore_mask, 0.0)
            labels = labels.masked_fill(ignore_mask, 0.0)

        loss = -torch.sum(logsigmoid(logits * labels), dim=-1).mean()

        diag_mean = torch.diag(logits).mean().item()
        ey = torch.eye(logits.size(-1), dtype=torch.bool, device=logits.device)
        off_mean = logits.masked_fill(ey, 0).mean().item()
        self.verbose and print("\nT=", T, "\nb=", b, "\ndiag", diag_mean, "\noff", off_mean, "\nloss=", loss, "\n")

        return loss


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


def info_nce(query, positive_key, negative_keys=None, temperature=0.1, reduction='mean', negative_mode='unpaired'):
    # Check input dimensionality.
    if query.dim() != 2:
        raise ValueError('<query> must have 2 dimensions.')
    if positive_key.dim() != 2:
        raise ValueError('<positive_key> must have 2 dimensions.')
    if negative_keys is not None:
        if negative_mode == 'unpaired' and negative_keys.dim() != 2:
            raise ValueError("<negative_keys> must have 2 dimensions if <negative_mode> == 'unpaired'.")
        if negative_mode == 'paired' and negative_keys.dim() != 3:
            raise ValueError("<negative_keys> must have 3 dimensions if <negative_mode> == 'paired'.")

    # Check matching number of samples.
    if len(query) != len(positive_key):
        raise ValueError('<query> and <positive_key> must must have the same number of samples.')
    if negative_keys is not None:
        if negative_mode == 'paired' and len(query) != len(negative_keys):
            raise ValueError(
                "If negative_mode == 'paired', then <negative_keys> must have the same number of samples as <query>.")

    # Embedding vectors should have same number of components.
    if query.shape[-1] != positive_key.shape[-1]:
        raise ValueError('Vectors of <query> and <positive_key> should have the same number of components.')
    if negative_keys is not None:
        if query.shape[-1] != negative_keys.shape[-1]:
            raise ValueError('Vectors of <query> and <negative_keys> should have the same number of components.')

    # Normalize to unit vectors
    query, positive_key, negative_keys = normalize(query, positive_key, negative_keys)
    if negative_keys is not None:
        # Explicit negative keys

        # Cosine between positive pairs
        positive_logit = torch.sum(query * positive_key, dim=1, keepdim=True)

        if negative_mode == 'unpaired':
            # Cosine between all query-negative combinations
            negative_logits = query @ transpose(negative_keys)

        elif negative_mode == 'paired':
            query = query.unsqueeze(1)
            negative_logits = query @ transpose(negative_keys)
            negative_logits = negative_logits.squeeze(1)

        # First index in last dimension are the positive samples
        logits = torch.cat([positive_logit, negative_logits], dim=1)
        labels = torch.zeros(len(logits), dtype=torch.long, device=query.device)
    else:
        # Negative keys are implicitly off-diagonal positive keys.

        # Cosine between all combinations
        logits = query @ transpose(positive_key)

        # Positive keys are the entries on the diagonal
        labels = torch.arange(len(query), device=query.device)

    return F.cross_entropy(logits / temperature, labels, reduction=reduction)


def transpose(x):
    return x.transpose(-2, -1)


def normalize(*xs):
    return [None if x is None else F.normalize(x, dim=-1) for x in xs]


def masked_info_nce_2d(za: Tensor, za_mask: Tensor, zb: Tensor, zb_mask: Tensor, tau: float = .05) \
        -> tuple[Tensor, int]:
    idx = (za_mask.bool() & zb_mask.bool()).nonzero(as_tuple=True)[0]
    if idx.numel() <= 1:
        return torch.tensor(.0, device=za.device), 0

    a = F.normalize(za[idx])
    b = F.normalize(zb[idx].detach())
    logits = (a @ b.T) / tau
    # Return the INFO NCE loss + valid number of rows for this loss calc. As loss is compared with others we have to reduce it
    # to a valid range [0,1]. This allows the loss to be better controlled while training.
    return F.cross_entropy(logits, torch.arange(idx.numel(), device=a.device)), idx.numel()


# TODO Probabilemnte erore di indexing>
def masked_cosine_kd(za: Tensor, za_mask: Tensor, zb: Tensor, zb_mask: Tensor, verbose=True) -> tuple[Tensor, int]:
    idx = (za_mask.bool().squeeze() & zb_mask.bool().squeeze()).nonzero(as_tuple=True)[0]
    if idx.numel() == 0:
        return torch.tensor(0.0, device=za.device), 0

    a = F.normalize(za[idx], p=2, dim=-1)
    b = F.normalize(zb[idx].detach(), p=2, dim=-1)

    # If a var across rows ≈ 0 (and/or max same-side sim (a)≈1), your student collapsed ⇒ fix training/initialization/mask (see §3).
    # If both sides have healthy variance, but cos diag mean ≈ off mean and kd_top1 ≈ 1/24, you’re misaligned (see §2).
    with torch.no_grad():
        # A) shape & valid count
        verbose and print("a,b shapes:", a.shape, b.shape)

        # B) norms (should be ~1 after F.normalize)
        verbose and verbose and print("||a|| mean:", a.norm(dim=-1).mean().item(), "std:", a.norm(dim=-1).std().item())
        verbose and print("||b|| mean:", b.norm(dim=-1).mean().item(), "std:", b.norm(dim=-1).std().item())

        # C) per-dimension variance across rows (detect collapse)
        verbose and print("a var across rows (mean over dims):", a.var(dim=0).mean().item())
        verbose and print("b var across rows (mean over dims):", b.var(dim=0).mean().item())

        # D) cosine matrix diag vs off-diag
        C = a @ b.T  # [n,n]
        diag = C.diag().mean().item()
        off = (C.sum() - C.diag().sum()) / (C.numel() - C.size(0))
        verbose and print("cos diag mean:", diag, "off mean:", off.item())

        # E) are rows identical-ish? (max pairwise sim among different rows)
        Ca = (a @ a.T) - torch.eye(a.size(0), device=a.device)
        Cb = (b @ b.T) - torch.eye(b.size(0), device=b.device)
        verbose and print("max same-side sim (a):", Ca.max().item(), " (b):", Cb.max().item())

        # Zero/near-zero vectors before normalization → after F.normalize they become (almost) zero. Count them
        # If high: check your pooling path. Are you averaging with almost empty masks? Guard with clamp_min(eps)
        # (you already do) and drop rows with den==0 rather than filling with zeros
        pre = za[idx]  # before F.normalize
        bad = (pre.norm(dim=-1) < 1e-8).float().mean().item()
        verbose and print("fraction near-zero student rows:", bad)

    # Cosine similarity loss (optimizes direction, not magnitude)
    loss = 1 - F.cosine_similarity(a, b, dim=-1).mean()

    logits = (a.detach() @ b.T.detach()).float()  # [n,n]
    top1 = logits.argmax(1)  # best teacher for each student
    diag_acc = (top1 == torch.arange(logits.size(0), device=logits.device)).float().mean()
    verbose and print("\nkd_top1", diag_acc)  # should climb → 1.0 on overfit
    perm = logits.argmax(1)  # [n]
    verbose and print("perm (first 10):", perm)

    logits_id = (a.detach() @ a.T.detach()).float()
    diag_acc = (logits_id.argmax(1) == torch.arange(a.size(0), device=a.device)).float().mean()
    assert diag_acc == 1.0, "KD pipeline reorders/duplicates inside student path"  # todo ha trhowato questo
    return loss, idx.numel()


# TODO Prova a vedere se con questo ok
# kd_top1 ≈ 1/24 + constant perm ⇒ either pairs misaligned or student collapsed
def kd_cosine(a, b, valid=None):
    # a,b: [N,D], valid: [N] bool or None (already aligned)
    a = F.normalize(a, dim=-1)
    b = F.normalize(b, dim=-1)
    if valid is not None:
        a = a[valid];
        b = b[valid]
    return (1 - (a * b).sum(-1)).mean()


def kd_info_nce_by_ids(za, ids_s, zb, ids_t, tau=0.07):
    pos_t = {int(k): j for j, k in enumerate(ids_t)}
    ia, ib = [], []
    for i, k in enumerate(ids_s):
        j = pos_t.get(int(k))
        if j is not None:
            ia.append(i);
            ib.append(j)
    ia = torch.tensor(ia, device=za.device);
    ib = torch.tensor(ib, device=za.device)
    a = F.normalize(za[ia], dim=-1)
    b = F.normalize(zb[ib].detach(), dim=-1)
    logits = (a @ b.T).float() / tau
    targets = torch.arange(logits.size(0), device=za.device)
    return F.cross_entropy(logits, targets, reduction="mean"), logits


def masked_mse_kd(za: Tensor, za_mask: Tensor, zb: Tensor, zb_mask: Tensor) -> tuple[Tensor, int]:
    idx = (za_mask.bool() & zb_mask.bool()).nonzero(as_tuple=True)[0]
    if idx.numel() == 0:
        return torch.tensor(0.0, device=za.device), 0

    a = F.normalize(za[idx], p=2, dim=-1)
    b = F.normalize(zb[idx].detach(), p=2, dim=-1)

    loss = F.mse_loss(a, b)
    return loss, idx.numel()


def masked_cosine_similarity(za: Tensor, zb: Tensor, present: Tensor):
    w = present.float()
    # Masked mean of (1 - cos)
    return ((1 - F.cosine_similarity(za, zb, dim=-1)) * w).sum() / w.sum().clamp_min(1.0)

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
    z = F.normalize(z, dim=-1)
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
    a = za
    b = zb
    with torch.no_grad():
        a_n = F.normalize(a, dim=-1)
        b_n = F.normalize(b, dim=-1)
        S = a_n @ b_n.T
        diag = S.diag()
        off = S[~torch.eye(S.size(0), dtype=torch.bool, device=S.device)]
        print("pos cos mean/std:", float(diag.mean()), float(diag.std()))
        print("neg cos mean/std:", float(off.mean()), float(off.std()))

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


def masked_info_nce_2d(za: Tensor, za_mask: Tensor, zb: Tensor, zb_mask: Tensor, tau: float = .07) -> tuple[
    Tensor, int]:
    idx = (za_mask.bool() & zb_mask.bool()).nonzero(as_tuple=True)[0]
    if idx.numel() <= 1:
        return torch.tensor(.0, device=za.device), 0

    a = F.normalize(za[idx], p=2, dim=-1)
    b = F.normalize(zb[idx].detach(), p=2, dim=-1)
    logits = (a @ b.T) / tau
    # Return the INFO NCE loss + valid number of rows for this loss calc. As loss is compared with others we have to reduce it
    # to a valid range [0,1]. This allows the loss to be better controlled while training.
    return F.cross_entropy(logits, torch.arange(idx.numel(), device=a.device)), idx.numel()


def masked_cosine_kd(za: Tensor, za_mask: Tensor, zb: Tensor, zb_mask: Tensor) -> tuple[Tensor, int]:
    idx = (za_mask.bool().squeeze() & zb_mask.bool().squeeze()).nonzero(as_tuple=True)[0]
    if idx.numel() == 0:
        return torch.tensor(0.0, device=za.device), 0

    a = F.normalize(za[idx], p=2, dim=-1)
    b = F.normalize(zb[idx].detach(), p=2, dim=-1)

    # Cosine similarity loss (optimizes direction, not magnitude)
    loss = 1 - F.cosine_similarity(a, b, dim=-1).mean()

    logits = (a.detach() @ b.T.detach()).float()  # [n,n]
    top1 = logits.argmax(1)  # best teacher for each student
    diag_acc = (top1 == torch.arange(logits.size(0), device=logits.device)).float().mean()
    print("\nkd_top1", diag_acc)  # should climb → 1.0 on overfit
    perm = logits.argmax(1)  # [n]
    print("perm (first 10):", perm[:10])

    return loss, idx.numel()


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

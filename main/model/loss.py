import torch
from torch import Tensor
from torch.nn.functional import normalize, logsigmoid, binary_cross_entropy_with_logits
import torch.nn.functional as F


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
    loss = -torch.sum(logsigmoid(logits * labels), dim=-1).mean()
    return loss


def masked_info_nce_2d(za: Tensor, za_mask: Tensor, zb: Tensor, zb_mask: Tensor, tau: float = .2) -> tuple[Tensor, int]:
    idx = (za_mask.bool() & zb_mask.bool()).nonzero(as_tuple=True)[0]
    if idx.numel() <= 1:
        return torch.tensor(.0, device=za.device), 0

    a = F.normalize(za[idx], p=2, dim=-1)
    b = F.normalize(zb[idx].detach(), p=2, dim=-1)
    logits = (a @ b.T) / tau
    # Return the INFO NCE loss + valid number of rows for this loss calc.
    return F.cross_entropy(logits, torch.arange(idx.numel(), device=a.device)), idx.numel()


def masked_cosine_similarity(za: Tensor, zb: Tensor, present: Tensor):
    w = present.float()
    # Masked mean of (1 - cos)
    return ((1 - F.cosine_similarity(za, zb, dim=-1)) * w).sum() / w.sum().clamp_min(1.0)


def diagnose_siglip(za: Tensor, zb: Tensor, logt: Tensor = None, bias: Tensor = None, verbose: bool = True):
    """
    Full diagnostic version of SigLIP to find what's wrong.
    Run this on your FIRST batch and share the output.
    """
    if logt is None:
        logt = torch.log(torch.tensor(10.0))
    if bias is None:
        bias = torch.tensor(-10.0)

    if verbose:
        print("\n" + "=" * 70)
        print("SIGLIP DIAGNOSTIC")
        print("=" * 70)

    # Step 1: Check inputs
    if verbose:
        print(f"\n1. INPUT SHAPES:")
        print(f"   za: {za.shape}")
        print(f"   zb: {zb.shape}")
        print(f"   logt: {logt.item():.4f} → T = {torch.exp(logt).item():.4f}")
        print(f"   bias: {bias.item():.4f}")

    T = torch.exp(logt).to(za.device)
    B = za.shape[0]

    # Step 2: Check normalization BEFORE normalize
    if verbose:
        print(f"\n2. NORMS BEFORE NORMALIZATION:")
        za_norms_before = za.norm(dim=-1)
        zb_norms_before = zb.norm(dim=-1)
        print(f"   za: min={za_norms_before.min():.3f}, max={za_norms_before.max():.3f}, mean={za_norms_before.mean():.3f}")
        print(f"   zb: min={zb_norms_before.min():.3f}, max={zb_norms_before.max():.3f}, mean={zb_norms_before.mean():.3f}")

        if za_norms_before.max() > 100:
            print(f"   ⚠️  WARNING: za has very large norms! This will cause huge logits.")
        if zb_norms_before.max() > 100:
            print(f"   ⚠️  WARNING: zb has very large norms! This will cause huge logits.")

    # L2-Normalization
    za = F.normalize(za, p=2, dim=-1)
    zb = F.normalize(zb, p=2, dim=-1)

    # Step 3: Check normalization AFTER
    if verbose:
        print(f"\n3. NORMS AFTER NORMALIZATION:")
        za_norms_after = za.norm(dim=-1)
        zb_norms_after = zb.norm(dim=-1)
        print(f"   za: min={za_norms_after.min():.3f}, max={za_norms_after.max():.3f}, mean={za_norms_after.mean():.3f}")
        print(f"   zb: min={zb_norms_after.min():.3f}, max={zb_norms_after.max():.3f}, mean={zb_norms_after.mean():.3f}")

        if not torch.allclose(za_norms_after, torch.ones_like(za_norms_after), atol=1e-5):
            print(f"   ⚠️  WARNING: za normalization failed!")
        if not torch.allclose(zb_norms_after, torch.ones_like(zb_norms_after), atol=1e-5):
            print(f"   ⚠️  WARNING: zb normalization failed!")

    # Step 4: Similarities (before temperature)
    similarities = za @ zb.T
    if verbose:
        print(f"\n4. SIMILARITIES (before temperature/bias):")
        print(f"   Range: [{similarities.min():.3f}, {similarities.max():.3f}]")
        pos_sims = torch.diagonal(similarities)
        neg_sims = similarities[~torch.eye(B, dtype=torch.bool, device=za.device)]
        print(f"   Positive (diagonal): mean={pos_sims.mean():.3f}, std={pos_sims.std():.3f}")
        print(f"   Negative (off-diag): mean={neg_sims.mean():.3f}, std={neg_sims.std():.3f}")
        print(f"   Gap: {pos_sims.mean() - neg_sims.mean():.3f}")

    # Step 5: Logits (after temperature and bias)
    logits = (za @ zb.T) * T + bias.to(za.device)

    if verbose:
        print(f"\n5. LOGITS (after temperature={T.item():.2f} and bias={bias.item():.2f}):")
        print(f"   Range: [{logits.min():.3f}, {logits.max():.3f}]")
        pos_logits = torch.diagonal(logits)
        neg_logits = logits[~torch.eye(B, dtype=torch.bool, device=za.device)]
        print(f"   Positive (diagonal): mean={pos_logits.mean():.3f}, std={pos_logits.std():.3f}")
        print(f"   Negative (off-diag): mean={neg_logits.mean():.3f}, std={neg_logits.std():.3f}")

        if logits.max() > 50:
            print(f"   ⚠️  WARNING: Logits are very large! This causes numerical issues.")
        if logits.min() < -50:
            print(f"   ⚠️  WARNING: Logits are very negative! This causes numerical issues.")

    # Step 6: Labels
    labels = 2 * torch.eye(B, device=za.device) - 1

    if verbose:
        print(f"\n6. LABELS:")
        print(f"   Shape: {labels.shape}")
        print(f"   Diagonal (should be +1): {labels.diagonal().unique().tolist()}")
        if B <= 5:
            print(f"   Full matrix:\n{labels}")

        # Check for the bug
        if not torch.allclose(labels.diagonal(), torch.ones(B, device=za.device)):
            print(f"   ❌ ERROR: Diagonal is not all +1!")
        off_diag = labels[~torch.eye(B, dtype=torch.bool, device=za.device)]
        if not torch.allclose(off_diag, -torch.ones_like(off_diag)):
            print(f"   ❌ ERROR: Off-diagonal is not all -1!")

    # Step 7: Loss terms
    loss_terms = -F.logsigmoid(logits * labels)

    if verbose:
        print(f"\n7. LOSS TERMS:")
        pos_loss = torch.diagonal(loss_terms)
        neg_loss = loss_terms[~torch.eye(B, dtype=torch.bool, device=za.device)]
        print(f"   Positive losses: mean={pos_loss.mean():.3f}, max={pos_loss.max():.3f}")
        print(f"   Negative losses: mean={neg_loss.mean():.3f}, max={neg_loss.max():.3f}")

        if pos_loss.mean() > 10:
            print(f"   ⚠️  WARNING: Positive losses very high! Positives are not matching.")
        if neg_loss.mean() > 10:
            print(f"   ⚠️  WARNING: Negative losses very high! Negatives are too similar.")

    # Step 8: Final loss
    loss = -torch.sum(F.logsigmoid(logits * labels), dim=-1).mean()

    if verbose:
        print(f"\n8. FINAL LOSS:")
        print(f"   Loss: {loss.item():.4f}")
        print(f"   Expected at random init: ~0.69")

        if loss.item() > 5.0:
            print(f"   ❌ LOSS TOO HIGH!")
            print(f"   Likely causes:")
            print(f"      - Embeddings not properly normalized")
            print(f"      - Temperature T={T.item():.2f} might be wrong")
            print(f"      - Initial embeddings very dissimilar")
        elif loss.item() < 0.3:
            print(f"   ⚠️  LOSS TOO LOW! Already converged?")
        else:
            print(f"   ✓ Loss is reasonable")

    if verbose:
        print("=" * 70 + "\n")

    return loss

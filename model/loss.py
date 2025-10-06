import torch
from torch import Tensor
from torch.nn.functional import normalize, logsigmoid, binary_cross_entropy_with_logits

# TODO: This is for sure a term of my loss but I have to balance for the EEG data.
#       I need to ensure that EEG stays relevant:
"""
# Contrastive learning on the cross-attention output
final_eeg_pooled = final_sequential_output.mean(dim=1)  # (1, 384)
L_eeg_specific = contrastive_loss(final_eeg_pooled, positive_pairs, negative_pairs)

# Encourage temporally adjacent EEG segments to be similar
eeg_t1, eeg_t2 = split_temporal_segments(final_sequential_output)
L_eeg_specific = MSE(eeg_t1, eeg_t2) + temporal_smoothness_penalty

# If you have EEG-specific labels (emotion, cognitive load, etc.)
eeg_task_pred = eeg_classifier(final_sequential_output.mean(dim=1))  # Pool over sequence
L_eeg_specific = CrossEntropy(eeg_task_pred, eeg_labels)

# Use the output that has been modulated by other modalities
final_eeg_repr = final_sequential_output  # (1, 85, 384)
eeg_reconstructed = eeg_decoder(final_eeg_repr)
L_eeg_specific = MSE(eeg_reconstructed, original_eeg_input)


# Assuming your EEG sequence represents time steps
# Split the 85-token sequence into two halves or overlapping windows
eeg_repr = final_sequential_output  # (1, 85, 384)

# Method A: Split sequence
eeg_first_half = eeg_repr[:, :42, :]   # (1, 42, 384)
eeg_second_half = eeg_repr[:, 43:, :]  # (1, 42, 384)
L_temporal = MSE(eeg_first_half.mean(dim=1), eeg_second_half.mean(dim=1))

# Method B: Adjacent time windows
eeg_windows_t = eeg_repr[:, :-1, :]    # (1, 84, 384)
eeg_windows_t1 = eeg_repr[:, 1:, :]    # (1, 84, 384)
L_temporal = MSE(eeg_windows_t, eeg_windows_t1)

# If L_eeg_specific IS the reconstruction loss
L_total = α * L_siglip + β * L_eeg_reconstruction

# If you want both reconstruction AND another EEG-specific task
L_eeg_reconstruction = MSE(reconstructed_eeg, original_eeg)
L_eeg_specific = temporal_consistency_loss  # or some other EEG task
L_total = α * L_siglip + β * L_eeg_specific + γ * L_eeg_reconstruction

# Phase 1: Establish EEG importance
L = α * L_siglip + β * L_eeg_reconstruction  # β high

# Phase 2: Encourage specialization  
L = α * L_siglip + β * L_eeg_reconstruction  # β decreases over time

# Phase 3: Pure multimodal learning
L = α * L_siglip  # β → 0
"""
def siglip(za: Tensor, zb: Tensor, logt: Tensor = torch.log(Tensor([10])), bias: Tensor = Tensor([-10])):
    """
    We will use this one.
    TODO: Non considera heterogeneous shapes
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

import torch
from torch import nn


# Multimodal Adaptation Gate (MAG)
# https://aclanthology.org/2020.acl-main.214.pdf
# Our data leverages video more, we try to make it the anchor.
# Later if we manage to get good texts transcripts we could revert to original.
class MAG3D(nn.Module):
    def __init__(self, anchor_dim: int, y_dim: int, z_dim: int,
                 beta_shift: float, dropout: float, eps: float = 1e-6):
        super(MAG3D, self).__init__()
        self.epsilon = eps
        self.W_y = nn.Linear(y_dim, anchor_dim)
        self.W_z = nn.Linear(z_dim, anchor_dim)

        # Create the projection to anchor dim space with concatenation of inputs
        self.y_sequential = nn.Sequential(nn.Linear(y_dim + anchor_dim, anchor_dim), nn.ReLU())
        self.z_sequential = nn.Sequential(nn.Linear(z_dim + anchor_dim, anchor_dim), nn.ReLU())

        self.beta_shift = beta_shift  # β is a hyperparameter selected through the cross-validation process
        self.norm = nn.LayerNorm(anchor_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, anchor, y, z):
        # X is the Anchor. Y, Z. What we are given are all embeddings.
        gating_y = self.y_sequential(torch.cat((y, anchor), dim=-1))  # Gating of y
        gating_z = self.z_sequential(torch.cat((z, anchor), dim=-1))  # Gating of z

        h_m: torch.Tensor = gating_y * self.W_y(y) + gating_z * self.W_z(z)
        anchor_norm: torch.Tensor = anchor.norm_qo(p=2, dim=-1, keepdim=True)
        hm_norm: torch.Tensor = h_m.norm(p=2, dim=-1, keepdim=True)

        hm_norm_ones: torch.Tensor = torch.ones(hm_norm.shape, requires_grad=True)
        # Avoid division by 0
        hm_norm = torch.where(hm_norm == 0, hm_norm_ones, hm_norm)
        threshold = (anchor_norm / (hm_norm + self.epsilon)) * self.beta_shift

        ones = torch.ones(threshold.shape, requires_grad=True)
        alpha = torch.min(threshold, ones).unsqueeze(-1)

        output = self.dropout(self.norm(alpha * h_m + anchor))
        return output


class MAG2D(nn.Module):
    def __init__(self, anchor_dim: int, y_dim: int, beta_shift: float, dropout: float, eps: float = 1e-6):
        super(MAG2D, self).__init__()
        self.epsilon = eps

        self.W_y = nn.Linear(y_dim, anchor_dim)
        # Create the projection to anchor dim space with concatenation of inputs
        self.y_sequential = nn.Sequential(nn.Linear(y_dim + anchor_dim, anchor_dim), nn.ReLU())

        self.beta_shift = beta_shift  # β is a hyperparameter selected through the cross-validation process
        self.norm = nn.LayerNorm(anchor_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, anchor: torch.Tensor, y: torch.Tensor):
        gating_y = self.y_sequential(torch.cat((y, anchor), dim=-1))  # Gating of y

        h_m: torch.Tensor = gating_y * self.W_y(y)
        anchor_norm: torch.Tensor = anchor.norm(p=2, dim=-1, keepdim=True)
        hm_norm: torch.Tensor = h_m.norm(p=2, dim=-1, keepdim=True)

        hm_norm_ones: torch.Tensor = torch.ones(hm_norm.shape, requires_grad=True)
        # Avoid division by 0
        hm_norm = torch.where(hm_norm == 0, hm_norm_ones, hm_norm)
        threshold = (anchor_norm / (hm_norm + self.epsilon)) * self.beta_shift

        ones = torch.ones(threshold.shape, requires_grad=True)
        alpha = torch.min(threshold, ones).unsqueeze(-1)

        output = self.dropout(self.norm(alpha * h_m + anchor))
        return output

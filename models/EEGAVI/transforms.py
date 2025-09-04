import torch


def media_locs_single_item(B, Tq, device):
    m = torch.zeros(B, Tq, dtype=torch.bool, device=device)
    m[:, 0] = True  # Item “introduced” at t=0
    return m

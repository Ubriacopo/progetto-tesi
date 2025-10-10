import torch


def freeze_module(m: torch.nn.Module):
    for p in m.parameters():
        p.requires_grad = False



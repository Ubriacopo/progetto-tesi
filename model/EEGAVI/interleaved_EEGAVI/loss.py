import torch

from model.loss import siglip


# TODO definire
def loss(y, y_kd: dict[str, torch.Tensor], y_teacher: dict[str, torch.Tensor]):
    loss = .0
    # Mi sa che è sbagliato questo. Devo are plain distance.
    for key, s_value in y_kd.items():
        t_value = y_teacher[key]
        loss += siglip(s_value, t_value)

    return loss

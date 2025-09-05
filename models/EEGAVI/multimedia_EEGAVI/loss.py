import torch

from common.model.loss import sigLIP


# TODO definire
def loss(y, y_kd: dict[str, torch.Tensor], y_teacher: dict[str, torch.Tensor]):
    loss = .0
    for key, s_value in y_kd.items():
        t_value = y_teacher[key]
        loss += sigLIP(s_value, t_value)

    return loss

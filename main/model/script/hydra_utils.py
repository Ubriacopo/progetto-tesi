import dataclasses

import torch
from hydra.utils import get_object
from torch import nn

from main.core_data.dataset import RequiredKey
from main.model.VATE.constrastive_model import MaskedContrastiveModel
from main.model.neegavi.factory import Factory
from main.model.neegavi.model import EegInterAviModel
from main.model.script.hydra_beans import KdConfig


@dataclasses.dataclass
class InitReturn:
    student: nn.Module
    student_keys: list[RequiredKey]

    teacher: nn.Module
    teacher_keys: list[RequiredKey]

    fusion_metric_codes: list[str]


def init_trainlike_script(cfg: KdConfig):
    # cfg = OmegaConf.to_container(cfg, resolve=True)
    factory_constructor = get_object(cfg.model.factory.factory_path)
    factory: Factory = factory_constructor(**cfg.model.factory.args)
    student: EegInterAviModel = factory.build()

    # Teacher construction
    teacher = MaskedContrastiveModel(hidden_channels=cfg.teacher.hidden_channels, out_channels=cfg.teacher.out_channels)
    teacher.load_state_dict(torch.load(cfg.teacher_weights_path))
    teacher.eval()  # Set to evaluation mode as we won't be learning on teacher.

    fusion_metrics_codes = [cfg.model.supports[s].code for s in cfg.model.supports]
    fusion_metrics_codes.append(cfg.model.pivot.code)

    c = cfg.model.pivot
    student_keys: list[RequiredKey] = [RequiredKey(c.code, c.shape, c.mask_shape, c.cannot_miss)]
    teacher_keys: list[RequiredKey] = []
    if c.is_teacher_key:
        teacher_keys = [RequiredKey(c.code, c.teacher_shape, c.teacher_mask_shape, c.cannot_miss)]

    # Each support has to be registered as key in student and also at times in teacher.
    for key in cfg.model.supports:
        c = cfg.model.supports[key]
        student_keys.append(RequiredKey(c.code, c.shape, c.mask_shape, c.cannot_miss))
        if c.is_teacher_key:
            teacher_keys.append(RequiredKey(c.code, c.teacher_shape, c.teacher_mask_shape, c.cannot_miss))

    return InitReturn(
        student=student, student_keys=student_keys,
        teacher=teacher, teacher_keys=teacher_keys,
        fusion_metric_codes=fusion_metrics_codes
    )

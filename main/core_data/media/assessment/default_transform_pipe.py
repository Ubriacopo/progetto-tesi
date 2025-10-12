import torch
from torch import nn
from torchvision.transforms import v2

from main.core_data.media.assessment.assessment import Assessment


def assessment_transform_pipe():
    return Assessment.modality_code(), nn.Sequential(
        v2.Lambda(lambda x: torch.tensor(x.data))
    )

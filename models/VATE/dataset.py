from __future__ import annotations

import torch
from torch import nn
from torchvision.transforms import v2
from transformers import VivitImageProcessor, VivitForVideoClassification

from common.data.data_point import EEGModalityComposeWrapper


class CallViViTModelAndProcessor(nn.Module):
    def __init__(self, model: str = "google/vivit-b-16x2-kinetics400"):
        super().__init__()
        self.processor = VivitImageProcessor.from_pretrained(model)
        self.model = VivitForVideoClassification.from_pretrained(model)

    def __call__(self, video: list[torch.Tensor] | torch.Tensor, **kwargs) -> dict[str, torch.Tensor]:
        # TODO: Vedi se veramente così
        if isinstance(video, torch.Tensor) and len(video.shape) == 3:
            video = [video]
        elif isinstance(video, torch.Tensor):
            video = list(video.unbind(0))

        item = self.processor(video, return_tensors="pt")
        with torch.no_grad():
            item = self.model(**item).logits.squeeze(0)

        return item  # The item has been embedded.


class FaceCrop(nn.Module):
    # Always replicate the training pipeline end-to-end -> FACE crop on videos to VATE.
    # TODO: Implement (Per ora posticipo in quanto già face videos)
    def forward(self, video):
        return video


vate_transform = EEGModalityComposeWrapper(
    vid_transform=nn.Sequential(
        FaceCrop(),
        CallViViTModelAndProcessor(),
    )
)

if __name__ == "__main__":
    from torch import nn
    from common.data.amigos.transform import train_eeg_transform
    from common.data.amigos.transform import train_video_transform
    from common.data.amigos.transform import train_audio_transform
    from common.data.data_point import EEGModalityComposeWrapper
    from common.data.amigos.dataset import KDAmigosDataset

    dataset = KDAmigosDataset(
        dataset_spec_file="./../../resources/AMIGOS/processed/spec.csv",
        shared_transform=EEGModalityComposeWrapper(
            train_eeg_transform(),
            train_video_transform(),
            train_audio_transform(),
        ),
        modality_transforms=[
            EEGModalityComposeWrapper(
                vid_transform=nn.Sequential(
                    v2.Resize((128, 128)),
                    v2.ToDtype(torch.float32, scale=True),
                ),
            ),
            vate_transform
        ]
    )

    a = dataset[0]
    print("a")

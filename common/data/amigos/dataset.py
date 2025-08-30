import dataclasses

import torch
from transformers import VivitForVideoClassification, VivitImageProcessor

from common.data.amigos.transform import train_video_transform, train_audio_transform, train_eeg_transform
from common.data.data_point import EEGDatasetDataPoint
from common.data.dataset import EEGPdSpecMediaDataset, KDEEGPdSpecMediaDataset
from common.data.transform import KwargsCompose


class AMIGOSDataset(EEGPdSpecMediaDataset):
    pass

class KDAmigosDataset(KDEEGPdSpecMediaDataset):
    pass

@dataclasses.dataclass
class VATEViViTEmbedder:
    processor = VivitImageProcessor.from_pretrained("google/vivit-b-16x2-kinetics400")
    model = VivitForVideoClassification.from_pretrained("google/vivit-b-16x2-kinetics400")

    def __call__(self, x: torch.Tensor, *args, **kwargs):
        x = self.processor(x.unbind(0), return_tensors="pt")
        with torch.no_grad():
            x = self.model(**x).logits.squeeze(0)

        return x


def video_transform(grayscale_p):
    return KwargsCompose([VATEViViTEmbedder()])


class VateAdaptorTransform:
    def __init__(self, custom_transform: KwargsCompose = video_transform(0.1)):
        self.custom_transform = custom_transform

    # Prende input da un EEGPdSpecMediaDataset e lo trasforma come gli serve
    def __call__(self, x: EEGDatasetDataPoint, *args, **kwargs):
        x.vid = self.custom_transform(x.vid[0])
        return x


# Just to see it work todo move to experiemnts or remvoe
if __name__ == "__main__":
    dataset = KDAmigosDataset(
        dataset_spec_file="../../../resources/AMIGOS/processed/spec.csv",
        eeg_transform=[train_eeg_transform()],
        video_transform=[train_video_transform(),video_transform(0.1)],
        audio_transform=[train_audio_transform()],
    )

    dataset[0]
    print("a")

"""
    dataset = AMIGOSDataset(
        dataset_spec_file="../../../resources/AMIGOS/processed/spec.csv",
        eeg_transform=train_eeg_transform(),
        video_transform=train_video_transform(),
        audio_transform=train_audio_transform()
    )

    vate_mapper = VateAdaptorTransform()

    res = vate_mapper(dataset[0])
    print(res)
"""
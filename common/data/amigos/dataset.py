from torch import nn
from torchvision.transforms import Resize

from common.data.amigos.transform import train_video_transform, train_audio_transform, train_eeg_transform
from common.data.data_point import EEGDatasetTransformWrapper
from common.data.dataset import EEGPdSpecMediaDataset, KDEEGPdSpecMediaDataset


# todo non ha vero motivo di esistere
class AMIGOSDataset(EEGPdSpecMediaDataset):
    pass


class KDAmigosDataset(KDEEGPdSpecMediaDataset):
    pass


# Just to see it work todo move to experiemnts or remvoe
if __name__ == "__main__":
    dataset = KDAmigosDataset(
        dataset_spec_file="../../../resources/AMIGOS/processed/spec.csv",
        shared_transform=EEGDatasetTransformWrapper(
            train_eeg_transform(),
            train_video_transform(),
            train_audio_transform(),
        ),
        modality_transforms=[
            EEGDatasetTransformWrapper(
                vid_transform=nn.Sequential(
                    Resize((128, 128))
                ),
            ),
            EEGDatasetTransformWrapper(
                vid_transform=nn.Sequential(
                    Resize((56, 56))
                )
            ),
        ]
    )

    o = dataset[0]
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

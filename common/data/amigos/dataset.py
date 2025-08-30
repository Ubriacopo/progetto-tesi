from torchvision.transforms import v2

from common.data.amigos.transform import train_video_transform, train_audio_transform
from common.data.dataset import EEGPdSpecMediaDataset
from common.data.transform import KwargsCompose


class AMIGOSDataset(EEGPdSpecMediaDataset):
    pass


if __name__ == "__main__":
    dataset = AMIGOSDataset(
        dataset_spec_file="../../../resources/AMIGOS/processed/spec.csv",
        eeg_transform=KwargsCompose([
            v2.Lambda(lambda x: x.to("cuda"))  # Change device
        ]),
        video_transform=train_video_transform(),
        audio_transform=train_audio_transform()
    )
    o = dataset[0]
    i = dataset[15]
    print(o)
    print(i)

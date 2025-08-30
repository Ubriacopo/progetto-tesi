from torchvision.transforms import v2

import common.data.eeg.transforms as eegtfs
from common.data.amigos.transform import train_video_transform
from common.data.dataset import EEGPdSpecMediaDataset
from common.data.transform import KwargsCompose
from common.data.video.transforms import RegularFrameResampling


class AMIGOSDataset(EEGPdSpecMediaDataset):
    pass


if __name__ == "__main__":
    dataset = AMIGOSDataset(
        dataset_spec_file="../../../resources/AMIGOS/processed/spec.csv",
        eeg_transform=KwargsCompose([
            v2.Lambda(lambda x: x.to("cuda"))  # Change device
        ]),
        video_transform=train_video_transform(),
        audio_transform=KwargsCompose([])
    )
    o = dataset[0]
    i = dataset[15]
    print(o)
    print(i)

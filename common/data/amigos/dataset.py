from common.data.amigos.transform import train_video_transform, train_audio_transform, train_eeg_transform
from common.data.dataset import EEGPdSpecMediaDataset


class AMIGOSDataset(EEGPdSpecMediaDataset):
    pass

# Just to see it work todo move to experiemnts or remvoe
if __name__ == "__main__":
    dataset = AMIGOSDataset(
        dataset_spec_file="../../../resources/AMIGOS/processed/spec.csv",
        eeg_transform=train_eeg_transform(),
        video_transform=train_video_transform(),
        audio_transform=train_audio_transform()
    )
    o = dataset[0]
    i = dataset[15]
    print(o)
    print(i)

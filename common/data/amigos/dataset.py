from torchvision.transforms import v2

from common.data.dataset import EEGPdSpecMediaDataset
from common.data.transform import Compose
import common.data.eeg.transforms as eegtfs
from common.data.video.transforms import ResampleVideoFrames, ToVideoFileClip, VideoFileToTensor


class AMIGOSDataset(EEGPdSpecMediaDataset):
    pass


if __name__ == "__main__":
    dataset = AMIGOSDataset(
        dataset_spec_file="../../../resources/AMIGOS/processed/spec.csv",
        eeg_transform=Compose([
            eegtfs.MneToTensor(),
            v2.Lambda(lambda x: x.to("cuda"))  # Change device
        ]),
        video_transform=Compose([
            # todo: una callback loadmediafrompathspec (VIDEO ha uno, AUDIO ha uno), toTesnor -> poi lavoriamo solo su tensori
            #       Penso sia più pulito. Dataset diventà più lightweight perchè permette di caricare. + se manca la fn a testa la aggiungo io (facciamosubclassing)
            ResampleVideoFrames((25, 16)),
        ]),
        audio_transform=Compose([])
    )
    o = dataset[0]
    i = dataset[15]
    print(o)
    print(i)

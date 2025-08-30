from torchvision.transforms import v2

from common.data.dataset import EEGPdSpecMediaDataset
from common.data.transform import Compose, KwargsCompose
import common.data.eeg.transforms as eegtfs
from common.data.video.transforms import ResampleFrames
from common.data.video.video import RegularFrameResampling


class AMIGOSDataset(EEGPdSpecMediaDataset):
    pass


if __name__ == "__main__":
    dataset = AMIGOSDataset(
        dataset_spec_file="../../../resources/AMIGOS/processed/spec.csv",
        eeg_transform=KwargsCompose([
            eegtfs.MneToTensor(),
            v2.Lambda(lambda x: x.to("cuda"))  # Change device
        ]),
        video_transform=KwargsCompose([
            # todo: una callback loadmediafrompathspec (VIDEO ha uno, AUDIO ha uno), toTesnor -> poi lavoriamo solo su tensori
            #       Penso sia più pulito. Dataset diventà più lightweight perchè permette di caricare. + se manca la fn a testa la aggiungo io (facciamosubclassing)
            RegularFrameResampling(32),
        ]),
        audio_transform=KwargsCompose([])
    )
    o = dataset[0]
    i = dataset[15]
    print(o)
    print(i)

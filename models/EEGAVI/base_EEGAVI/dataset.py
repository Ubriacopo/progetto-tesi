from torch.utils.data import StackDataset
from torchaudio.transforms import Resample
from torchvision.transforms import Normalize

from common.data.amigos.config import CH_NAMES, CH_TYPES
from common.data.audio.transforms import AudioZeroMasking, AudioToTensor, ToMono
from common.data.data_point import EEGDatasetTransformWrapper
from common.data.dataset import KDEEGPdSpecMediaDataset
from common.data.eeg.transforms import EEGToMneRawFromChannels, EEGResample, EEGToTensor, EEGToTimePatches
from common.data.video import VideoToTensor, RegularFrameResampling
from models.FEEG.transforms import W2VBertFeatureExtractorTransform
from common.data.video.transforms import ViVitImageProcessorTransform
from models.VATE.dataset import VATE_AMIGOS_transforms


def kd_train_dataset(amigos_path: str):
    return StackDataset(
        KDEEGPdSpecMediaDataset(
            amigos_path,
            shared_transform=EEGDatasetTransformWrapper(
                name="shared_transform",
                vid_transform=[
                    VideoToTensor(),
                    # RGB normalization. VIVIT was trained with this so we go for it. (Both VATE and EEGAVI use vivit as backbone)
                    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                    # As we don't have T we cannot pad the media.
                    # We still downsample if the video is too long but pad else in case of T
                    RegularFrameResampling(32, drop_mask=True),
                ],
                aud_transform=[
                    AudioToTensor(),
                    ToMono(),
                ],
                eeg_transform=[
                    EEGToMneRawFromChannels(CH_NAMES, CH_TYPES),
                    EEGResample(200, 128),
                    EEGToTensor(),
                    EEGToTimePatches(200),
                ]
            ),
            # todo usa arg così le definisco per ds da altre parti
            modality_transforms=[
                EEGDatasetTransformWrapper(
                    name="EEGAVI",
                    vid_transform=[
                        ViVitImageProcessorTransform(),
                    ],
                    aud_transform=[
                        Resample(44000, 16000),
                        AudioZeroMasking(8, 16000),
                        W2VBertFeatureExtractorTransform(),
                    ]
                ),
                VATE_AMIGOS_transforms()
            ]
        )
    )

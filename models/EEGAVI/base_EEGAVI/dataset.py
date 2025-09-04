from torch.utils.data import StackDataset
from torchaudio.transforms import Resample

from common.data.amigos.config import CH_NAMES, CH_TYPES
from common.data.audio.transforms import AudioZeroMasking, AudioToTensor, ToMono
from common.data.data_point import EEGDatasetTransformWrapper
from common.data.dataset import KDEEGPdSpecMediaDataset
from common.data.eeg.transforms import EEGToMneRawFromChannels, EEGResample, EEGToTensor, EEGToTimePatches
from common.data.video import VideoToTensor, RegularFrameResampling
from models.FEEG.transforms import ViVitImageProcessorTransform, W2VBertFeatureExtractorTransform


def kd_train_dataset(amigos_path: str):
    return StackDataset(
        KDEEGPdSpecMediaDataset(
            amigos_path,
            shared_transform=EEGDatasetTransformWrapper(
                name="shared_transform",
                vid_transform=[
                    VideoToTensor(),
                    # As we don't have T we cannot pad the media.
                    # We still downsample if the video is too long but pad else in case of T
                    RegularFrameResampling(32, drop_mask=True),
                ],
                aud_transform=[
                    AudioToTensor(),
                    Resample(44000, 16000),
                    AudioZeroMasking(8, 16000),
                    ToMono(),
                ],
                eeg_transform=[
                    EEGToMneRawFromChannels(CH_NAMES, CH_TYPES),
                    EEGResample(200, 128),
                    EEGToTensor(),
                    EEGToTimePatches(200),
                ]
            ),
            modality_transforms=[
                EEGDatasetTransformWrapper(
                    name="EEGAVI",
                    vid_transform=[
                        ViVitImageProcessorTransform(),
                    ],
                    aud_transform=[
                        W2VBertFeatureExtractorTransform(),
                    ]
                ),
                EEGDatasetTransformWrapper(
                    name="VATEKD"
                )
            ]
        )
    )

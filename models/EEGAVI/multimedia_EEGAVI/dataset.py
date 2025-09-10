from torch.utils.data import StackDataset
from torchaudio.transforms import Resample
from torchvision.transforms import v2

from common.data.audio.transforms import AudioToTensor, AudioZeroMasking, ToMono
from common.data.data_point import EEGDatasetTransformWrapper
from common.data.dataset import KDEEGPdSpecMediaDataset
from common.data.eeg.transforms import EEGToMneRawFromChannels, EEGResample, EEGToTensor, EEGToTimePatches
from common.data.video import VideoToTensor, RegularFrameResampling
from models.FEEG.transforms import W2VBertFeatureExtractorTransform
from common.data.video.transforms import ViVitImageProcessorTransform


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
                        ViVitImageProcessorTransform(force_time_seq=True),
                    ],
                    aud_transform=[
                        W2VBertFeatureExtractorTransform(force_time_seq=True),
                    ],
                    eeg_transform=[
                        v2.Lambda(lambda x: x.unsqueeze(0)),
                    ]
                ),
                EEGDatasetTransformWrapper(
                    name="VATEKD"
                )
            ],
        )
    )

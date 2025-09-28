from torch.utils.data import StackDataset
from torchaudio.transforms import Resample
from torchvision.transforms import v2

from core_data.media.audio.transforms import AudioToTensor, ToMono
from common.data.signal.transforms import SignalZeroMasking
from core_data.data_point import EEGDatasetTransformWrapper
from core_data.dataset import KDEEGPdSpecMediaDataset
from core_data.media.eeg import EEGToMneRaw, EEGResample, EEGToTensor, EEGToTimePatches
from core_data.media.video import VideoToTensor, RegularFrameResampling
from model.FEEG.transforms import W2VBertFeatureExtractorTransform
from core_data.media.video import ViVitImageProcessorTransform


# todo riscrivere
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
                    SignalZeroMasking(8, 16000),
                    ToMono(),
                ],
                eeg_transform=[
                    EEGToMneRaw(CH_NAMES, CH_TYPES),
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

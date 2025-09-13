import torch
from torch.utils.data import StackDataset
from torchaudio.transforms import Resample
from torchvision.transforms import Normalize, v2
from torchvision.transforms.v2 import ToDtype
from transformers import VivitImageProcessor

from common.data.amigos.config import AmigosConfig
from common.data.audio.transforms import AudioZeroMasking, AudioToTensor, ToMono
from common.data.data_point import EEGDatasetTransformWrapper
from common.data.dataset import KDEEGPdSpecMediaDataset
from common.data.eeg.transforms import EEGToMneRaw, EEGResample, EEGToTensor, EEGToTimePatches
from common.data.video import VideoToTensor, RegularFrameResampling
from common.data.video.transforms import ViVitImageProcessorTransform
from models.FEEG.transforms import W2VBertFeatureExtractorTransform
from models.VATE.dataset import VATE_AMIGOS_transforms


def get_ViVit_processor():
    processor: VivitImageProcessor = VivitImageProcessor.from_pretrained("google/vivit-b-16x2-kinetics400")
    # Prepare ViVit for our task. Works as intended.
    processor.do_resize = True
    processor.do_rescale = True
    processor.do_normalize = True

    return processor


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
                    # TODO: What type does vivit want?
                    # ToDtype(torch.float32, scale=True),
                    # RGB normalization. VIVIT was trained with this so we go for it. (Both VATE and EEGAVI use vivit as backbone)
                    # Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ],
                aud_transform=[
                    AudioToTensor(),
                    ToMono(),
                ],
                eeg_transform=[
                    EEGToMneRaw(AmigosConfig.CH_NAMES, AmigosConfig.CH_TYPES),
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
                        # TODO Normalize rgb in questi. Sono fatti apposta
                        ViVitImageProcessorTransform(
                            # TODO Move the model name to config somewhere
                            processor=get_ViVit_processor()
                        ),
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

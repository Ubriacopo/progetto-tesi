from __future__ import annotations

from torch import nn
from torchvision.transforms import v2

from main.core_data.data_point import FlexibleDatasetTransformWrapper
from main.core_data.media.assessment.assessment import Assessment
from main.core_data.media.assessment.transform import SliceAssessments, ToTensorData, \
    PermuteAssessments
from main.core_data.media.audio import AudTargetConfig
from main.core_data.media.audio.default_transform_pipe import aud_wav2vec_interleaved_txt_extract_transform_pipe, \
    aud_vate_basic_transform_pipe
from main.core_data.media.ecg import EcgTargetConfig
from main.core_data.media.ecg.default_transform_pipe import ecg_interleaved_transform_pipe
from main.core_data.media.eeg.config import EegTargetConfig
from main.core_data.media.eeg.default_transform_pipe import eeg_transform_pipe
from main.core_data.media.metadata.metadata import Metadata
from main.core_data.media.metadata.transforms import MetadataToTensor
from main.core_data.media.text import Text
from main.core_data.media.text import TxtTargetConfig
from main.core_data.media.text.default_transform_pipe import txt_from_aud_interleaved_txt_extract_transform_pipe, \
    txt_vate_basic_transform_pipe
from main.core_data.media.text.transforms import RestoreTextExtract
from main.core_data.media.video import VidTargetConfig
from main.core_data.media.video.default_transform_pipe import vid_vivit_interleaved_transform_pipe, \
    vid_vate_basic_transform_pipe
from main.core_data.processing.preprocessing import TorchExportsSegmentsReadyPreprocessor
from main.dataset.amigos.config import AmigosConfig
from main.utils.args import safe_call


def assessment_transform_pipe():
    return Assessment.modality_code(), nn.Sequential(
        SliceAssessments(max_idx=4),  # Pick first 4 dimensions that we know are correct
        ToTensorData(),  # Go for tensor structure
        PermuteAssessments(original_order="a v d l")  # Sort them to canonical order. Used notation like einops.
    )


# todo refactor
@safe_call
def amigos_interleaved_preprocessor(
        output_max_length: int, output_path: str,
        extraction_data_folder: str,
        eeg_config: EegTargetConfig,
        ecg_config: EcgTargetConfig,
        aud_config: AudTargetConfig,
        vid_config: VidTargetConfig,
        txt_config: TxtTargetConfig
):
    config = AmigosConfig(
        aud_target_config=aud_config,
        vid_target_config=vid_config,
        txt_target_config=txt_config,
        ecg_target_config=ecg_config,
        eeg_target_config=eeg_config,
        max_length=output_max_length
    )

    return TorchExportsSegmentsReadyPreprocessor(
        output_path=output_path,
        segment_pipeline=FlexibleDatasetTransformWrapper(
            "interleaved_preprocessor",
            aud_wav2vec_interleaved_txt_extract_transform_pipe(config),
            vid_vivit_interleaved_transform_pipe(config),
            eeg_transform_pipe(config),
            ecg_interleaved_transform_pipe(config),
            txt_from_aud_interleaved_txt_extract_transform_pipe(config),
            (Assessment.modality_code(), nn.Sequential(v2.Lambda(lambda x: x.data))),
            (Metadata.modality_code(), MetadataToTensor())
        ),
        sample_pipeline=FlexibleDatasetTransformWrapper(
            "shared_interleaved_preprocessor",
            (Text.modality_code(), RestoreTextExtract(base_path=extraction_data_folder)),
            assessment_transform_pipe(),
        ),
        extraction_data_folder=extraction_data_folder
    )


# todo add index to output (cosi da verificare corretto allineamnto tra i ds)
@safe_call
def amigos_vate_basic_preprocessor(output_path: str, extraction_data_folder: str,
                                   vid_config: VidTargetConfig = VidTargetConfig(), ):
    config = AmigosConfig(vid_target_config=vid_config, )
    return TorchExportsSegmentsReadyPreprocessor(
        output_path=output_path,
        extraction_data_folder=extraction_data_folder,
        segment_pipeline=FlexibleDatasetTransformWrapper(
            "default_preprocessor",
            vid_vate_basic_transform_pipe(config),
            aud_vate_basic_transform_pipe(config),
            txt_vate_basic_transform_pipe(),
            (Metadata.modality_code(), MetadataToTensor())
        ),
        sample_pipeline=FlexibleDatasetTransformWrapper(
            "shared_interleaved_preprocessor",
            (Text.modality_code(), RestoreTextExtract(base_path=extraction_data_folder))
        ),
    )

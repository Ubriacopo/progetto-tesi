from __future__ import annotations

import torch
from einops.layers.torch import Rearrange
from torch import nn
from torchvision.transforms import v2

from main.core_data.data_point import FlexibleDatasetTransformWrapper
from main.core_data.media.assessment.assessment import Assessment
from main.core_data.media.assessment.transform import RemapFieldToRange, SliceAssessments, ToTensorData, \
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
    return TorchExportsSegmentsReadyPreprocessor(
        output_path=output_path,
        segment_pipeline=FlexibleDatasetTransformWrapper(
            "interleaved_preprocessor",
            aud_wav2vec_interleaved_txt_extract_transform_pipe(
                aud_config, AmigosConfig.Audio.fs, output_max_length
            ),
            vid_vivit_interleaved_transform_pipe(vid_config, AmigosConfig.Video.fps, output_max_length),
            eeg_transform_pipe(target_config=eeg_config, eeg_order=AmigosConfig.EEG_CHANNELS,
                               source_fs=AmigosConfig.EEG.fs, max_length=output_max_length),
            # TODO a 1s come video
            ecg_interleaved_transform_pipe(ecg_config, AmigosConfig.EEG.fs, output_max_length),
            # TODO a 1s come video
            txt_from_aud_interleaved_txt_extract_transform_pipe(txt_config, output_max_length),
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
    return TorchExportsSegmentsReadyPreprocessor(
        output_path=output_path,
        extraction_data_folder=extraction_data_folder,
        segment_pipeline=FlexibleDatasetTransformWrapper(
            "default_preprocessor",
            vid_vate_basic_transform_pipe(vid_config),
            aud_vate_basic_transform_pipe(AmigosConfig.Audio.fs),
            txt_vate_basic_transform_pipe(),
            (Metadata.modality_code(), MetadataToTensor())
        ),
        sample_pipeline=FlexibleDatasetTransformWrapper(
            "shared_interleaved_preprocessor",
            (Text.modality_code(), RestoreTextExtract(base_path=extraction_data_folder))
        ),
    )

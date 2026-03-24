from __future__ import annotations

from torch import nn
from torchvision.transforms import v2

from main.core_data.data_point import FlexibleDatasetTransformWrapper
from main.core_data.media.assessment.assessment import Assessment
from main.core_data.media.assessment.transform import AssessmentToTensor
from main.core_data.media.audio.default_transform_pipe import aud_wav2vec_interleaved_txt_extract_transform_pipe, \
    aud_vate_basic_transform_pipe
from main.core_data.media.ecg.default_transform_pipe import ecg_interleaved_transform_pipe
from main.core_data.media.eeg.default_transform_pipe import eeg_transform_pipe, eeg_sample_pipeline
from main.core_data.media.metadata.metadata import Metadata
from main.core_data.media.metadata.transforms import MetadataToTensor
from main.core_data.media.text import Text
from main.core_data.media.text.default_transform_pipe import txt_from_aud_interleaved_txt_extract_transform_pipe, \
    txt_vate_basic_transform_pipe
from main.core_data.media.text.transforms import RestoreTextExtract
from main.core_data.media.video.default_transform_pipe import vid_vivit_interleaved_transform_pipe, \
    vid_vate_basic_transform_pipe
from main.core_data.processing.preprocessing import TorchExportsSegmentsReadyPreprocessor, \
    TorchExportsKdSegmentsReadyPreprocessor
from main.core_data.sampler import FixedIntervalsSegmenter
from main.dataset.amigos.config import AmigosConfig


def interleaved_preprocessor(output_path: str, extraction_data_folder: str, config: AmigosConfig):
    return TorchExportsSegmentsReadyPreprocessor(
        output_path=output_path,
        extraction_data_folder=extraction_data_folder,
        segment_pipeline=FlexibleDatasetTransformWrapper(
            "interleaved_preprocessor",
            eeg_transform_pipe(config),
            aud_wav2vec_interleaved_txt_extract_transform_pipe(config),
            vid_vivit_interleaved_transform_pipe(config),
            ecg_interleaved_transform_pipe(config),
            txt_from_aud_interleaved_txt_extract_transform_pipe(config),
            (Assessment.modality_code(), nn.Sequential(v2.Lambda(lambda x: x.data))),
            (Metadata.modality_code(), MetadataToTensor())
        ),
        sample_pipeline=FlexibleDatasetTransformWrapper(
            "shared_interleaved_preprocessor",
            (Text.modality_code(), RestoreTextExtract(base_path=extraction_data_folder)),
            # assessment_transform_pipe(),
            eeg_sample_pipeline(config)
        ),
    )


def interleaved_downstream_preprocessor(output_path: str, extraction_data_folder: str, config: AmigosConfig):
    return TorchExportsSegmentsReadyPreprocessor(
        output_path=output_path,
        # extraction_data_folder=extraction_data_folder,
        segment_pipeline=FlexibleDatasetTransformWrapper(
            "segment_interleaved_downstream_preprocessor",
            eeg_transform_pipe(config),
            aud_wav2vec_interleaved_txt_extract_transform_pipe(config),
            vid_vivit_interleaved_transform_pipe(config),
            ecg_interleaved_transform_pipe(config),
            txt_from_aud_interleaved_txt_extract_transform_pipe(config),
            (Assessment.modality_code(), AssessmentToTensor()),
            (Metadata.modality_code(), MetadataToTensor())
        ),
        sample_pipeline=FlexibleDatasetTransformWrapper(
            "shared_interleaved_downstream_preprocessor",
            (Text.modality_code(), RestoreTextExtract(base_path=extraction_data_folder)),
            eeg_sample_pipeline(config)
        ),
        segmenter=FixedIntervalsSegmenter(12)
    )


def vate_preprocessor(output_path: str, extraction_data_folder: str, config: AmigosConfig):
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


def combined_preprocessor(output_path: str, extraction_data_folder: str, config: AmigosConfig):
    return TorchExportsKdSegmentsReadyPreprocessor(
        output_path=output_path,
        extraction_data_folder=extraction_data_folder,
        student_segment_pipeline=FlexibleDatasetTransformWrapper(
            "AMIGOS_interleaved_preprocessor",
            aud_wav2vec_interleaved_txt_extract_transform_pipe(config),
            vid_vivit_interleaved_transform_pipe(config),
            eeg_transform_pipe(config),
            ecg_interleaved_transform_pipe(config),
            txt_from_aud_interleaved_txt_extract_transform_pipe(config),
            (Assessment.modality_code(), nn.Sequential(v2.Lambda(lambda x: x.data))),
            (Metadata.modality_code(), MetadataToTensor())
        ),
        student_sample_pipeline=FlexibleDatasetTransformWrapper(
            "AMIGOS_shared_interleaved_preprocessor",
            (Text.modality_code(), RestoreTextExtract(base_path=extraction_data_folder)),
            # assessment_transform_pipe(),
            eeg_sample_pipeline(config)
        ),
        teacher_segment_pipeline=FlexibleDatasetTransformWrapper(
            "AMIGOS_vate_default_preprocessor",
            vid_vate_basic_transform_pipe(config),
            aud_vate_basic_transform_pipe(config),
            txt_vate_basic_transform_pipe(),
            (Metadata.modality_code(), MetadataToTensor())
        ),

        teacher_sample_pipeline=FlexibleDatasetTransformWrapper(
            "shared_interleaved_preprocessor",
            (Text.modality_code(), RestoreTextExtract(base_path=extraction_data_folder))
        )
    )

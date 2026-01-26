from main.core_data.data_point import FlexibleDatasetTransformWrapper
from main.core_data.media.audio import Audio
from main.core_data.media.ecg import ECG
from main.core_data.media.ecg.default_transform_pipe import ecg_interleaved_transform_pipe
from main.core_data.media.eeg import EEG
from main.core_data.media.eeg.default_transform_pipe import eeg_transform_pipe, eeg_sample_pipeline
from main.core_data.media.metadata.metadata import Metadata
from main.core_data.media.metadata.transforms import MetadataToTensor
from main.core_data.media.text import Text
from main.core_data.media.video import Video
from main.core_data.media.video.default_transform_pipe import vid_vivit_interleaved_transform_pipe, \
    vid_vate_basic_transform_pipe
from main.core_data.processing.preprocessing import TorchExportsSegmentsReadyPreprocessor, \
    TorchExportsKdSegmentsReadyPreprocessor
from main.dataset.mahnob.config import MahnobConfig


def interleaved_preprocessor(output_path: str, extraction_data_folder: str, config: MahnobConfig):
    return TorchExportsSegmentsReadyPreprocessor(
        output_path=output_path,
        extraction_data_folder=extraction_data_folder,
        segment_pipeline=FlexibleDatasetTransformWrapper(
            "MANHOB-interleaved-processor",
            vid_vivit_interleaved_transform_pipe(config),
            eeg_transform_pipe(config),
            ecg_interleaved_transform_pipe(config),
            (Metadata.modality_code(), MetadataToTensor())
        ),
        sample_pipeline=FlexibleDatasetTransformWrapper(
            "MANHOB-sample-pipeline",
            eeg_sample_pipeline(config)
        )
    )


def vate_preprocessor(output_path: str, extraction_data_folder: str, config: MahnobConfig):
    return TorchExportsSegmentsReadyPreprocessor(
        output_path=output_path,
        extraction_data_folder=extraction_data_folder,
        segment_pipeline=FlexibleDatasetTransformWrapper(
            "MANHOB-vate-processor",
            vid_vate_basic_transform_pipe(config),
            (Metadata.modality_code(), MetadataToTensor())
        ),

    )


def combined_preprocessor(output_path: str, extraction_data_folder: str, config: MahnobConfig):
    return TorchExportsKdSegmentsReadyPreprocessor(
        output_path=output_path,
        extraction_data_folder=extraction_data_folder,
        student_segment_pipeline=FlexibleDatasetTransformWrapper(
            "MANHOB-interleaved-processor",
            vid_vivit_interleaved_transform_pipe(config),
            eeg_transform_pipe(config),
            ecg_interleaved_transform_pipe(config),
            (Metadata.modality_code(), MetadataToTensor())
        ),
        student_sample_pipeline=FlexibleDatasetTransformWrapper(
            "MANHOB-sample-pipeline",
            eeg_sample_pipeline(config)
        ),
        teacher_segment_pipeline=FlexibleDatasetTransformWrapper(
            "MANHOB-vate-processor",
            vid_vate_basic_transform_pipe(config),
            (Metadata.modality_code(), MetadataToTensor())
        ),
        quantization_keys=[
            Video.modality_code(),
            Audio.modality_code(),
            Text.modality_code(),
            EEG.modality_code(),
            ECG.modality_code(),
        ]
    )

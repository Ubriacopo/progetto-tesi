from main.core_data.data_point import FlexibleDatasetTransformWrapper
from main.core_data.media.audio.default_transform_pipe import aud_wav2vec_interleaved_txt_extract_transform_pipe
from main.core_data.media.eeg.default_transform_pipe import eeg_sample_pipeline, eeg_transform_pipe
from main.core_data.media.metadata.metadata import Metadata
from main.core_data.media.metadata.transforms import MetadataToTensor
from main.core_data.media.text import Text
from main.core_data.media.text.default_transform_pipe import txt_from_aud_interleaved_txt_extract_transform_pipe
from main.core_data.media.text.transforms import RestoreTextExtract
from main.core_data.media.video.default_transform_pipe import vid_vivit_interleaved_transform_pipe
from main.core_data.processing.preprocessing import TorchExportsSegmentsReadyPreprocessor
from main.dataset.mmdapbe.config import MmdapbeConfig


# todo see what to keep but of these segment pipelines and what to add
def interleaved_preprocessor(output_path: str, extraction_data_folder: str, config: MmdapbeConfig):
    return TorchExportsSegmentsReadyPreprocessor(
        output_path=output_path,
        extraction_data_folder=extraction_data_folder,
        segment_pipeline=FlexibleDatasetTransformWrapper(
            "Mmdapbe-interleaved-processor",
            vid_vivit_interleaved_transform_pipe(config),
            aud_wav2vec_interleaved_txt_extract_transform_pipe(config),
            eeg_transform_pipe(config),
            txt_from_aud_interleaved_txt_extract_transform_pipe(config),
            (Metadata.modality_code(), MetadataToTensor())
        ),
        sample_pipeline=FlexibleDatasetTransformWrapper(
            "shared_interleaved_preprocessor",
            (Text.modality_code(), RestoreTextExtract(base_path=extraction_data_folder)),
            eeg_sample_pipeline(config)
        )
    )

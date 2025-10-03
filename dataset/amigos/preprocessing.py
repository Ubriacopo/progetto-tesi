from __future__ import annotations

from core_data.data_point import FlexibleDatasetTransformWrapper
from core_data.media.audio import AudTargetConfig
from core_data.media.audio import Audio
from core_data.media.audio.default_transform_pipe import aud_wav2vec_interleaved_txt_extract_transform_pipe, \
    aud_wav2vec_default_txt_extract_transform_pipe
from core_data.media.ecg import EcgTargetConfig
from core_data.media.ecg.default_transform_pipe import ecg_interleaved_transform_pipe, ecg_default_transform_pipe
from core_data.media.eeg.config import EegTargetConfig
from core_data.media.eeg.default_transform_pipe import eeg_transform_pipe
from core_data.media.text import Text
from core_data.media.text import TxtTargetConfig
from core_data.media.text.default_transform_pipe import txt_from_aud_interleaved_txt_extract_transform_pipe
from core_data.media.text.transforms import WhisperClipTextExtract, RestoreTextExtract
from core_data.media.video import VidTargetConfig
from core_data.media.video.default_transform_pipe import vid_vivit_interleaved_transform_pipe
from core_data.processing.preprocessing import TorchExportsSegmenterPreprocessor, TorchExportsSegmentsReadyPreprocessor
from core_data.sampler import EegFeaturesAndRandLogUIntervalsSegmenter, Segmenter
from dataset.amigos.config import AmigosConfig
from dataset.amigos.loader import AmigosPointsLoader


def amigos_interleaved_preprocessor(
        output_max_length: int, output_path: str,
        extraction_data_folder: str,
        aud_config: AudTargetConfig = AudTargetConfig(),
        vid_config: VidTargetConfig = VidTargetConfig(),
        txt_config: TxtTargetConfig = TxtTargetConfig("./gen-text-out.txt"),
        eeg_config: EegTargetConfig = EegTargetConfig("../../dependencies/cbramod/pretrained_weights.pth"),
        ecg_config: EcgTargetConfig = EcgTargetConfig(AmigosConfig.prepare_ecg),
):
    return TorchExportsSegmentsReadyPreprocessor(
        output_path=output_path,
        segment_pipeline=FlexibleDatasetTransformWrapper(
            "interleaved_preprocessor",
            aud_wav2vec_interleaved_txt_extract_transform_pipe(
                aud_config, AmigosConfig.Audio.fs, output_max_length
            ),
            vid_vivit_interleaved_transform_pipe(vid_config, AmigosConfig.Video.fps, output_max_length),
            eeg_transform_pipe(eeg_config, AmigosConfig.EEG.fs, output_max_length),
            ecg_interleaved_transform_pipe(ecg_config, AmigosConfig.EEG.fs, output_max_length),
            txt_from_aud_interleaved_txt_extract_transform_pipe(txt_config, output_max_length),
        ),
        sample_pipeline=FlexibleDatasetTransformWrapper(
            "shared_interleaved_preprocessor",
            (Text.modality_code(), RestoreTextExtract(base_path=extraction_data_folder))
        ),
        extraction_data_folder=extraction_data_folder
    )


def amigos_vate_preprocessor(
        output_max_length: int, output_path: str,
        extraction_data_folder: str,
):
    return TorchExportsSegmentsReadyPreprocessor(
        output_path=output_path,
        extraction_data_folder=extraction_data_folder,
        segment_pipeline=FlexibleDatasetTransformWrapper(
            "default_preprocessor",
            # vid_vivit_default_transform_pipe(vid_config, AmigosConfig.Video.fps, output_max_length),
            # aud_wav2vec_default_txt_extract_transform_pipe(aud_config, AmigosConfig.Audio.fs, output_max_length),
            expand_nested=True,
            nested_keys=[Text.modality_code(), Audio.modality_code()],
        ),
        sample_pipeline=FlexibleDatasetTransformWrapper(
            "shared_interleaved_preprocessor",
            (Text.modality_code(), RestoreTextExtract(base_path=extraction_data_folder))
        ),
    )


if __name__ == "__main__":
    # Test the methods:
    interleaved_processor = amigos_interleaved_preprocessor(32, "../../../resources/AMIGOS/p-interleaved/")
    interleaved_processor.run(AmigosPointsLoader("../../../resources/AMIGOS/"))

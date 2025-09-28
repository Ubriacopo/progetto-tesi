from __future__ import annotations

from common.data.amigos.config import AmigosConfig
from common.data.amigos.loader import AmigosPointsLoader
from common.data.audio import Audio
from common.data.audio.config import AudTargetConfig
from common.data.audio.default_transform_pipe import aud_wav2vec_interleaved_txt_extract_transform_pipe, \
    aud_wav2vec_default_txt_extract_transform_pipe
from common.data.data_point import FlexibleDatasetTransformWrapper
from common.data.ecg.config import EcgTargetConfig
from common.data.ecg.default_transform_pipe import ecg_interleaved_transform_pipe, ecg_default_transform_pipe
from common.data.eeg.config import EegTargetConfig
from common.data.eeg.default_transform_pipe import eeg_transform_pipe
from common.data.preprocessing import TorchExportsSegmenterPreprocessor
from common.data.sampler import EegFeaturesAndRandLogUIntervalsSegmenter, Segmenter
from common.data.text import Text
from common.data.text.config import TxtTargetConfig
from common.data.text.default_transform_pipe import txt_from_aud_interleaved_txt_extract_transform_pipe, \
    shared_txt_transform_pipe
from common.data.text.transforms import WhisperClipTextExtract
from common.data.video.config import VidTargetConfig
from common.data.video.default_transform_pipe import vid_vivit_interleaved_transform_pipe, \
    vid_vivit_default_transform_pipe


def amigos_interleaved_preprocessor(
        output_max_length: int, output_path: str,
        segmenter: Segmenter = EegFeaturesAndRandLogUIntervalsSegmenter(
            min_length=2, max_length=32, num_segments=20, anchor_identification_hop=0.125, extraction_jitter=0.1
        ),
        aud_config: AudTargetConfig = AudTargetConfig(),
        vid_config: VidTargetConfig = VidTargetConfig(),
        txt_config: TxtTargetConfig = TxtTargetConfig("./gen-text-out.txt"),
        eeg_config: EegTargetConfig = EegTargetConfig("../../../dependencies/cbramod/pretrained_weights.pth"),
        ecg_config: EcgTargetConfig = EcgTargetConfig(AmigosConfig.prepare_ecg),
):
    return TorchExportsSegmenterPreprocessor(
        output_path=output_path,
        ch_names=AmigosConfig.CH_NAMES,
        ch_types=AmigosConfig.CH_TYPES,
        segmenter=segmenter,
        segment_pipeline=FlexibleDatasetTransformWrapper(
            "interleaved_preprocessor",
            aud_wav2vec_interleaved_txt_extract_transform_pipe(
                aud_config, txt_config, AmigosConfig.Audio.fs, output_max_length
            ),
            vid_vivit_interleaved_transform_pipe(vid_config, AmigosConfig.Video.fps, output_max_length),
            eeg_transform_pipe(eeg_config, AmigosConfig.EEG.fs, output_max_length),
            ecg_interleaved_transform_pipe(ecg_config, AmigosConfig.EEG.fs, output_max_length),
            txt_from_aud_interleaved_txt_extract_transform_pipe(txt_config, output_max_length),
        ),
        sample_pipeline=FlexibleDatasetTransformWrapper(
            "shared_interleaved_preprocessor",
            (Text.modality_code(), WhisperClipTextExtract(device="cpu"))
        )
    )


def amigos_default_preprocessor(
        output_max_length: int, output_path: str,
        segmenter: Segmenter = EegFeaturesAndRandLogUIntervalsSegmenter(
            min_length=2, max_length=32, num_segments=20, anchor_identification_hop=0.125, extraction_jitter=0.1
        ),
        aud_config: AudTargetConfig = AudTargetConfig(),
        vid_config: VidTargetConfig = VidTargetConfig(),
        eeg_config: EegTargetConfig = EegTargetConfig("../../../dependencies/cbramod/pretrained_weights.pth"),
        ecg_config: EcgTargetConfig = EcgTargetConfig(AmigosConfig.prepare_ecg),
):
    return TorchExportsSegmenterPreprocessor(
        output_path=output_path,
        ch_names=AmigosConfig.CH_NAMES,
        ch_types=AmigosConfig.CH_TYPES,
        segmenter=segmenter,
        segment_pipeline=FlexibleDatasetTransformWrapper(
            "default_preprocessor",
            vid_vivit_default_transform_pipe(vid_config, AmigosConfig.Video.fps, output_max_length),
            eeg_transform_pipe(eeg_config, AmigosConfig.EEG.fs, output_max_length),
            ecg_default_transform_pipe(ecg_config),
            aud_wav2vec_default_txt_extract_transform_pipe(aud_config, AmigosConfig.Audio.fs, output_max_length),
            expand_nested=True,
            nested_keys=[Text.modality_code(), Audio.modality_code()],
        ),
        sample_pipeline=FlexibleDatasetTransformWrapper(
            "shared_interleaved_preprocessor",
            (Text.modality_code(), WhisperClipTextExtract(device="cpu"))
        )
    )


if __name__ == "__main__":
    # Test the methods:
    interleaved_processor = amigos_interleaved_preprocessor(32, "../../../resources/AMIGOS/p-interleaved/")
    interleaved_processor.run(AmigosPointsLoader("../../../resources/AMIGOS/"))

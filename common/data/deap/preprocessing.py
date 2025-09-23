from abc import ABC, abstractmethod
from typing import Callable

from torch import nn
from torchvision.transforms import v2

from common.data.amigos.config import AmigosConfig
from common.data.audio import Audio
from common.data.audio.transforms import AudioZeroMasking
from common.data.data_point import AgnosticDatasetTransformWrapper
from common.data.deap.config import DeapConfig
from common.data.deap.loader import DeapPointsLoader
from common.data.ecg.ecg import ECG
from common.data.ecg.transforms import EcgDataAsTensor, EcgSequenceResampling, EcgFmEmbedderTransform
from common.data.eeg import EEG
from common.data.eeg.transforms import AddMneAnnotation, EEGResample, EEGToTimePatches, CBraModEmbedderTransform
from common.data.preprocessing import TorchExportsSegmenterPreprocessor
from common.data.sampler import FeatureAndRandomLogUniformIntervalsSegmenter
from common.data.signal.transforms import SubclipMneRaw, SignalToTensor
from common.data.transform import MultimediaPadding
from common.data.video import Video
from common.data.video.transforms import UnbufferedResize, SubclipVideo, VideoToTensor, ViVitImageProcessorTransform, \
    VideoSequenceResampling, RegularFrameResampling, ViVitEmbedderTransform


class BaseTransformFlowGenerator(ABC):
    @abstractmethod
    def transform(self) -> tuple[str, nn.Sequential]:
        pass


class VidMultiSequenceTransformGenerator(BaseTransformFlowGenerator):
    MAX_FPS = 32  # Limit set by ViVit

    def __init__(self, max_length: int, original_fps: int, sub_media_mas_duration: int):
        self.original_fps: int = original_fps
        self.sub_media_mas_duration: int = sub_media_mas_duration
        self.padding: int = int(max_length / self.sub_media_mas_duration)

    def transform(self) -> tuple[str, nn.Sequential]:
        return Video.modality_code(), nn.Sequential(
            SubclipVideo(),
            VideoToTensor(),
            ViVitImageProcessorTransform(),
            v2.Lambda(lambda x: x.pixel_values),
            VideoSequenceResampling(
                original_fps=self.original_fps, sequence_duration_seconds=self.sub_media_mas_duration,
                frames_resampler=RegularFrameResampling(self.MAX_FPS, drop_mask=True)
            ),
            # TODO: find a way to run in GPU without saturating memory -> fare il to e poi inverse! -> Infer veloce.
            #       -> se non passo tutta la sequenza in una botta sola funziona altrimenti no. -> Versione "memory safe".
            ViVitEmbedderTransform(device="cpu"),
            MultimediaPadding(max_length=self.padding),
        )


class EEGTransformFlowGenerator(BaseTransformFlowGenerator):
    def __init__(self, original_fs: int, target_fs: int, cbra_weights_path: str, max_segments: int):
        self.original_fs: int = original_fs
        self.target_fs: int = target_fs
        self.cbra_weights_path: str = cbra_weights_path
        self.max_segments: int = max_segments

    def transform(self) -> tuple[str, nn.Sequential]:
        return EEG.modality_code(), nn.Sequential(
            SubclipMneRaw(),
            EEGResample(tfreq=self.target_fs, sfreq=self.original_fs),
            SignalToTensor(),
            EEGToTimePatches(points_per_patch=self.target_fs, max_segments=self.max_segments),
            CBraModEmbedderTransform(weights_path=self.cbra_weights_path)
        )


class EcgTransformFlowGenerator(BaseTransformFlowGenerator):
    def __init__(self, original_fs: int, target_fs: int, sequence_duration_seconds: int,
                 prepare_ecg: Callable, ecg_fm_endpoint: str):
        self.original_fs: int = original_fs
        self.target_fs: int = target_fs  # todo 128 credo
        self.prepare_ecg = prepare_ecg  # = AmigosConfig.prepare_ecg
        self.sequence_duration_seconds: int = sequence_duration_seconds  # todo set to 6
        self.ecg_fm_endpoint: str = ecg_fm_endpoint

    def transform(self) -> tuple[str, nn.Sequential]:
        return ECG.modality_code(), nn.Sequential(
            SubclipMneRaw(),
            EcgDataAsTensor(),
            EcgSequenceResampling(
                # todo add masking ??
                original_fs=self.original_fs,
                sequence_duration_seconds=self.sequence_duration_seconds,
                # todo rename
                resampler=AudioZeroMasking(max_length=self.sequence_duration_seconds, fs=self.target_fs),
                channels_first=True,
            ),
            EcgFmEmbedderTransform(data_transform_fn=self.prepare_ecg, endpoint=self.ecg_fm_endpoint),
        )


# TODO Finish
def deap_interleaved_preprocessor(output_path: str,
                                  ecg_fm_endpoint: str = "localhost:7860/extract_features",
                                  cbramod_weights_path: str = "../../../dependencies/cbramod/pretrained_weights.pth",
                                  ) -> TorchExportsSegmenterPreprocessor:
    vid_gen = VidMultiSequenceTransformGenerator(
        max_length=30, original_fps=DeapConfig.original_fps, sub_media_mas_duration=4
    )

    eeg_gen = EEGTransformFlowGenerator(
        original_fs=DeapConfig.original_eeg_fs,
        target_fs=200,
        max_segments=10,
        cbra_weights_path=cbramod_weights_path
    )

    ecg_gen = EcgTransformFlowGenerator(
        ecg_fm_endpoint=ecg_fm_endpoint,
        original_fs=DeapConfig.original_eeg_fs,
        target_fs=128,
        sequence_duration_seconds=6,
        prepare_ecg=AmigosConfig.prepare_ecg
    )

    return TorchExportsSegmenterPreprocessor(
        output_path=output_path,
        ch_names=DeapConfig.CH_NAMES,
        ch_types=DeapConfig.CH_TYPES,
        segmenter=FeatureAndRandomLogUniformIntervalsSegmenter(
            min_length=2, max_length=32, num_segments=20, anchor_identification_hop=0.125, extraction_jitter=0.1
        ),
        pipeline=AgnosticDatasetTransformWrapper(
            "interleaved_preprocessor",
            vid_gen.transform(),
            eeg_gen.transform(),
            ecg_gen.transform(),
            # Audio and text do not exist so we don't use them.
        )
    )


class DeapPreprocessorFactory:
    @staticmethod
    def run_default(input_path: str, output_path: str, max_length: int = 8):
        return DeapPreprocessorFactory.default(output_path, max_length).run(DeapPointsLoader(input_path))

    @staticmethod
    def interleaved(output_path: str,
                    cbramod_weights_path: str = "../../../dependencies/cbramod/pretrained_weights.pth",
                    max_length: int = 30,
                    sub_media_max_length_seconds: int = 2,
                    ecg_endpoint: str = "localhost:7860/extract_features") -> TorchExportsSegmenterPreprocessor:
        vid_transform = nn.Sequential()
        return TorchExportsSegmenterPreprocessor(
            output_path=output_path,
            ch_names=DeapConfig.CH_NAMES,
            ch_types=DeapConfig.CH_TYPES,
            segmenter=FeatureAndRandomLogUniformIntervalsSegmenter(
                min_length=2, max_length=32, num_segments=20, anchor_identification_hop=0.125, extraction_jitter=0.1
            ),
            pipeline=AgnosticDatasetTransformWrapper(
                "interleaved",
                (Video.modality_code(), vid_transform)
            )
        )

    @staticmethod
    def default(output_path: str, max_length: int = 8):
        return TorchExportsSegmenterPreprocessor(
            output_path=output_path,
            ch_names=DeapConfig.CH_NAMES,
            ch_types=DeapConfig.CH_TYPES,
            segmenter=FeatureAndRandomLogUniformIntervalsSegmenter(
                min_length=2, max_length=32, num_segments=20, anchor_identification_hop=0.125, extraction_jitter=0.1
            ),
            pipeline=AgnosticDatasetTransformWrapper(
                "preprocessing-default",
                (
                    Video.modality_code(),
                    nn.Sequential(
                        UnbufferedResize((260, 260)),
                        SubclipVideo(),
                    )
                ),
                (
                    EEG.modality_code(),
                    nn.Sequential(
                        AddMneAnnotation()
                    ),
                ),
                (
                    Audio.modality_code(),
                    nn.Sequential()
                )
            ),
        )

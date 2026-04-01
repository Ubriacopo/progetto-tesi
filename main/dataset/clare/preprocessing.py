import math

from torch import nn

from main.core_data.data_point import FlexibleDatasetTransformWrapper
from main.core_data.media.assessment.assessment import Assessment
from main.core_data.media.assessment.transform import AssessmentToTensor
from main.core_data.media.ecg import ECG
from main.core_data.media.ecg.default_transform_pipe import ecg_interleaved_transform_pipe
from main.core_data.media.ecg.transforms import EcgSequenceResampling, EcgFmEmbedderTransform
from main.core_data.media.eeg import EEG
from main.core_data.media.eeg.default_transform_pipe import eeg_sample_pipeline, eeg_transform_pipe
from main.core_data.media.eeg.transforms import EegTimePadding, CBraModEmbedderTransform, CanonicalOrderTransform, \
    EEGToTimePatches, EEGResample
from main.core_data.media.metadata.metadata import Metadata
from main.core_data.media.metadata.transforms import MetadataToTensor
from main.core_data.media.signal.transforms import SignalToTensor, BandpassFilter, SubclipMneRaw, SignalZeroMasking
from main.core_data.processing.preprocessing import TorchExportsSegmentsReadyPreprocessor, ExperimentPreprocessor
from main.core_data.processing.transform import DataQuantizationTransform, SequentialWithFallback, MultimediaPadding, \
    EmptyQuantizedObjectTransform
from main.core_data.sampler import FixedIntervalsSegmenter
from main.dataset.clare.config import ClareConfig


def interleaved_downstream_preprocessor(output_path: str, extraction_data_folder: str, config: ClareConfig):
    # Todo sarebbe uyn altro preprocessor, non segemntato
    max_length = math.ceil(config.max_length / config.unit_seconds)
    latent_size, patches = 256, 32
    return ExperimentPreprocessor(
        output_path=output_path,
        pipeline=FlexibleDatasetTransformWrapper(
            "shared_interleaved_downstream_preprocessor",
            (ECG.modality_code(), SequentialWithFallback(
                EcgSequenceResampling(
                    channels_first=True,
                    sequence_duration_seconds=int(config.unit_seconds),
                    resampler=SignalZeroMasking(max_length=config.unit_seconds, fs=config.ecg_target_config.fs),
                ),
                EcgFmEmbedderTransform(
                    data_transform_fn=config.ecg_source_config.prepare_ecg,
                    endpoint=config.ecg_target_config.fm_endpoint
                ),
                MultimediaPadding(max_length=max_length),
                DataQuantizationTransform(),

                default_remap=EmptyQuantizedObjectTransform(
                    shape=(max_length, patches, latent_size), mask_shape=(max_length,)
                ),
            )),
            (EEG.modality_code(), nn.Sequential(
                BandpassFilter(l_freq=0.5, h_freq=40.0, notch=50.0),
                EEGResample(tfreq=config.eeg_target_config.fs, sfreq=config.eeg_source_config.fs),
                SignalToTensor(),
                # Because we have fs=200 and CBraMod wants fs as points per patch max_segments=max_length
                EEGToTimePatches(points_per_patch=config.eeg_target_config.fs, max_segments=config.max_length),
                CanonicalOrderTransform(eeg_order=config.eeg_source_config.EEG_CHANNELS),
                CBraModEmbedderTransform(weights_path=config.eeg_target_config.model_weights_path),
                EegTimePadding(max_length=config.max_length),
                DataQuantizationTransform()
            )),

            (Assessment.modality_code(), AssessmentToTensor(),),
            (Metadata.modality_code(), MetadataToTensor())
        ),
    )

def old_interleaved_downstream_preprocessor(output_path: str, extraction_data_folder: str, config: ClareConfig):
    # Todo sarebbe uyn altro preprocessor, non segemntato
    return TorchExportsSegmentsReadyPreprocessor(
        output_path=output_path,
        segment_pipeline=FlexibleDatasetTransformWrapper(
            "segment_interleaved_downstream_preprocessor",
            eeg_transform_pipe(config),
            ecg_interleaved_transform_pipe(config),
            (Assessment.modality_code(), AssessmentToTensor(),),
            (Metadata.modality_code(), MetadataToTensor())
        ),
        sample_pipeline=FlexibleDatasetTransformWrapper(
            "shared_interleaved_downstream_preprocessor",
            eeg_sample_pipeline(config)
        ),
        segmenter=FixedIntervalsSegmenter(10)
    )

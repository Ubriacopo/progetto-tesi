from torch import nn

from main.core_data.data_point import FlexibleDatasetTransformWrapper
from main.core_data.media.assessment.assessment import Dominance, Valence, Arousal
from main.core_data.media.eeg.default_transform_pipe import eeg_transform_pipe, eeg_sample_pipeline
from main.core_data.media.metadata.metadata import Metadata
from main.core_data.media.metadata.transforms import MetadataToTensor
from main.core_data.processing.preprocessing import TorchExportsSegmentsReadyPreprocessor
from main.core_data.sampler import FixedIntervalsSegmenter
from main.dataset.dreamer.config import DreamerConfig


def interleaved_preprocessor(output_path: str, config: DreamerConfig):
    return TorchExportsSegmentsReadyPreprocessor(
        output_path=output_path,
        segment_pipeline=FlexibleDatasetTransformWrapper(
            "dreamer-interleaved-preprocessor",
            eeg_transform_pipe(config),
            # Per carita funziona ma poco elegante.
            # Cosi ho per ogni 16s il vettore della label, che andrà bene direi
            (Arousal.modality_code(), nn.Identity()),
            (Valence.modality_code(), nn.Identity()),
            (Dominance.modality_code(), nn.Identity()),

            (Metadata.modality_code(), MetadataToTensor())
        ),
        sample_pipeline=FlexibleDatasetTransformWrapper(
            "reamer-vate-sample-pipeline",
            eeg_sample_pipeline(config)
        ),
        segmenter=FixedIntervalsSegmenter(max_length=16, num_segments=1)
    )

from torch import nn

from main.core_data.data_point import FlexibleDatasetTransformWrapper
from main.core_data.media.assessment.assessment import Assessment, AssessmentLabels
from main.core_data.media.assessment.transform import RescaleAssessmentValue, AssessmentToTensor, TrackNanAssessment
from main.core_data.media.eeg.default_transform_pipe import eeg_transform_pipe, eeg_sample_pipeline
from main.core_data.media.metadata.metadata import Metadata
from main.core_data.media.metadata.transforms import MetadataToTensor
from main.core_data.processing.preprocessing import TorchExportsSegmentsReadyPreprocessor
from main.core_data.sampler import FixedIntervalsSegmenter
from main.dataset.dreamer.config import DreamerConfig


def interleaved_downstream_preprocessor(output_path: str, out, config: DreamerConfig, *args, **kwargs):
    return TorchExportsSegmentsReadyPreprocessor(
        output_path=output_path,
        segment_pipeline=FlexibleDatasetTransformWrapper(
            "dreamer-interleaved-preprocessor",
            eeg_transform_pipe(config),
            # Per carita funziona ma poco elegante.
            # Cosi ho per ogni 16s il vettore della label, che andrà bene direi
            (
                Assessment.modality_code(),
                nn.Sequential(
                    RescaleAssessmentValue(AssessmentLabels.AROUSAL, (1., 9.)),
                    RescaleAssessmentValue(AssessmentLabels.VALENCE, (1., 9.)),
                    RescaleAssessmentValue(AssessmentLabels.DOMINANCE, (1., 9.)),
                    TrackNanAssessment(AssessmentLabels.LIKING, rescale_range=(1., 9.)),
                    TrackNanAssessment(AssessmentLabels.FAMILIARITY, rescale_range=(1., 9.)),
                    AssessmentToTensor()
                ),

            ),
            (Metadata.modality_code(), MetadataToTensor())
        ),
        sample_pipeline=FlexibleDatasetTransformWrapper(
            "reamer-vate-sample-pipeline",
            eeg_sample_pipeline(config)
        ),
        segmenter=FixedIntervalsSegmenter(max_length=12)
    )

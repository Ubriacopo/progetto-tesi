from __future__ import annotations

from common.data.amigos.loader import AMIGOSLoader
from common.data.data_point_transforms import ResizeEEGDataPointMedia, SubclipMedia
from common.data.eeg.transforms import EEGMneAddAnnotation
from common.data.preprocessing import EEGSegmenterPreprocessor
from common.data.sampler import Segmenter, FixedIntervalsSegmenter
from common.data.transform import Compose


class AMIGOSPreprocessor(EEGSegmenterPreprocessor):
    @staticmethod
    def execute_default(input_path: str, output_path: str, max_length: int = 8):
        return AMIGOSPreprocessor.default(output_path, max_length).run(AMIGOSLoader(input_path))

    @staticmethod
    def default(output_path: str, max_length: int = 8) -> AMIGOSPreprocessor:
        return AMIGOSPreprocessor(
            output_path=output_path,
            segmenter=FixedIntervalsSegmenter(max_length),
            sample_pipeline=None,
            # todo magari dividere per modality? cosi sara piu pulito
            split_pipeline=Compose([
                SubclipMedia(),
                EEGMneAddAnnotation(),
                ResizeEEGDataPointMedia((260, 260))
            ])
        )

    def __init__(self, output_path: str, segmenter: Segmenter,
                 sample_pipeline: Compose = None, split_pipeline: Compose = None):
        ch_names = [
            "AF3", "F7", "F3", "FC5", "T7", "P7", "O1", "O2", "P8", "T8", "FC6", "F4", "F8", "AF4",  # EEG Channels
            "ECG Right", "ECG Left", "GSR"  # Others (ECG + ECG + MISC)
        ]

        ch_types = ["eeg"] * 14 + ["ecg"] * 2 + ["misc"]
        super().__init__(output_path, segmenter, ch_names, ch_types, sample_pipeline, split_pipeline)


if __name__ == "__main__":
    AMIGOSPreprocessor.execute_default(
        "../../../resources/AMIGOS/",
        "../../../resources/AMIGOS/processed/"
    )

from pathlib import Path

import numpy as np
from scipy.io import loadmat

from common.data.dataset import EEGDataset
from common.data.media import NumpyDataMediaCollector, FileReferenceMediaCollector, PandasCsvDataMediaCollector
from common.data.preprocessing import MediaPreProcessingPipeline


class AnotherAMIGOSDataset(EEGDataset):
    def __init__(self, signal_processor: MediaPreProcessingPipeline,
                 video_processor: MediaPreProcessingPipeline,
                 audio_processor: MediaPreProcessingPipeline,
                 text_processor: MediaPreProcessingPipeline, base_path: str):
        super().__init__(
            NumpyDataMediaCollector([], signal_processor),
            FileReferenceMediaCollector(video_processor),
            FileReferenceMediaCollector(audio_processor),
            PandasCsvDataMediaCollector([], text_processor),
            base_path
        )

    def scan(self):
        processed_data = Path(self.base_path + "pre_processed_py/")
        # if not processed_data.exists():
            # for f in Path(processed_data).iterdir():
                # extract_trial_data(self.base_path + "pre_processed_py/", str(f))

        # load_participant_data(Path(self.base_path + "pre_processed_py/"))

        metadata_folder = self.base_path + "metadata/"

        # Video files name: PXX_ (Person number)
        face_video_folder = self.base_path + "face/"
        face_folder = Path(face_video_folder)

        for v in face_folder.iterdir():
            pass

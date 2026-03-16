from dataclasses import asdict
from pathlib import Path

import mne.io
import numpy as np
import scipy

from main.core_data.data_point import FlexibleDatasetPoint
from main.core_data.loader import DataPointsLoader
from main.core_data.media.assessment.assessment import Assessment
from main.core_data.media.eeg import EEG
from main.core_data.media.metadata.metadata import Metadata, MetaObject
from main.dataset.faced.config import FacedConfig
from main.dataset.utils import DatasetUidStore


class FacedPointsLoader(DataPointsLoader):
    DATASET_ID: int = 6

    def __init__(self, base_path: str, dataset_uid_store: DatasetUidStore, config: FacedConfig = FacedConfig()):
        super().__init__(dataset_uid_store)
        self.base_path: Path = Path(base_path)
        self.config: FacedConfig = config
        self.length: int = 0

    def __len__(self) -> int:
        if self.length == 0:
            folder = Path(self.base_path / "Processed_data")
            self.length = sum(1 for _ in folder.iterdir()) * 28  # 28 experiments per person

        return self.length

    def scan(self):
        for subject_records in (self.base_path / "Processed_data").iterdir():
            assessments = scipy.io.matlab.loadmat(self.base_path / "Data" / subject_records.stem / "After_remarks.mat")
            assessments = assessments["After_remark"]
            eeg = np.load(subject_records, allow_pickle=True)

            for experiment in range(eeg.shape[0]):
                nei = self.dataset_uid_store.uid(subject_records.stem, str(experiment), "faced")

                experiment_scores = assessments[experiment][0]["score"][0]
                experiment_eeg = eeg[experiment]

                info = mne.create_info(
                    ch_names=self.config.eeg_source_config.get_CH_NAMES(),
                    ch_types=self.config.eeg_source_config.get_CH_TYPES(),
                    sfreq=self.config.eeg_source_config.fs
                )

                raw = mne.io.RawArray(experiment_eeg, info=info)
                person_id = int(subject_records.stem[3:])
                metadata = MetaObject(
                    experiment=nei, dataset_id=self.DATASET_ID, person_id=person_id, trial=experiment,
                )

                yield FlexibleDatasetPoint(
                    nei,
                    EEG(eid=nei, data=raw, fs=raw.info['sfreq']).as_mod_tuple(),
                    Assessment(eid=nei, data=experiment_scores, labels=self.config.score_labels.labels,
                               rating_scales=self.config.score_labels.rating_scales).as_mod_tuple(),
                    Metadata(data=asdict(metadata), eid=nei).as_mod_tuple()
                )

from dataclasses import asdict
from typing import Iterator, Optional

import mne
import numpy as np
from mne.io import RawArray
from scipy.io import loadmat

from main.core_data.data_point import FlexibleDatasetPoint
from main.core_data.loader import DataPointsLoader
from main.core_data.media.assessment.assessment import Assessment
from main.core_data.media.eeg import EEG
from main.core_data.media.metadata.metadata import MetaObject, Metadata
from main.dataset.dreamer.config import DreamerConfig
from main.dataset.utils import DatasetUidStore


class DreamerPointsLoader(DataPointsLoader):
    DATASET_ID: int = 5

    def __init__(self, base_path: str, dataset_uid_store: DatasetUidStore, config: DreamerConfig = DreamerConfig()):
        super().__init__(dataset_uid_store)
        self.base_path = base_path
        self.config: DreamerConfig = config
        self.length: Optional[int] = None

    def __len__(self) -> int:
        if self.length is not None:
            return self.length
        # 5 is index of n experiments 4 is index of participants
        self.length = 23 * 18
        return self.length

    # TODO finish
    def scan(self) -> Iterator[FlexibleDatasetPoint]:
        dreamer = loadmat(f"{self.base_path}/DREAMER.mat", simplify_cells=True)["DREAMER"]
        # Count number of experiments
        for user in range(len(dreamer["Data"])):
            # Thank you matlab
            user_experiment_data = dreamer["Data"][user]

            # Experiments with index
            for experiment_index in range(len(user_experiment_data["ScoreValence"])):
                nei = self.dataset_uid_store.uid(str(user), str(experiment_index), "DREAMER")
                # EEG Data: Stimuli is what matters to us. The normalization of bias is not of help? Guess not
                eeg_np = user_experiment_data["EEG"]["stimuli"][experiment_index].T  # Time x Channels
                info = mne.create_info(
                    ch_names=self.config.eeg_source_config.get_CH_NAMES(),
                    ch_types=self.config.eeg_source_config.get_CH_TYPES(),
                    sfreq=self.config.eeg_source_config.fs
                )
                raw: RawArray = mne.io.RawArray(eeg_np, info=info, verbose=False)
                valence = user_experiment_data["ScoreValence"][experiment_index]
                arousal = user_experiment_data["ScoreArousal"][experiment_index]
                dominance = user_experiment_data["ScoreDominance"][experiment_index]
                assessment = np.array([arousal, valence, dominance])

                metadata = MetaObject(
                    experiment=nei, dataset_id=self.DATASET_ID, person_id=user, trial=experiment_index
                )

                yield FlexibleDatasetPoint(
                    nei,
                    EEG(eid=nei, data=raw.copy().pick(["eeg"]), fs=raw.info['sfreq']).as_mod_tuple(),
                    # For evaluation todo rifai in stile altri
                    Assessment(
                        eid=nei,
                        data=assessment,
                        labels=self.config.score_labels_config.labels,
                        scales=self.config.score_labels_config.scales
                    ).as_mod_tuple(),
                    Metadata(data=asdict(metadata), eid=nei).as_mod_tuple()
                )

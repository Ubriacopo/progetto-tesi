from dataclasses import asdict
from typing import Iterator, Optional

import mne
from mne.io import RawArray
from scipy.io import loadmat

from main.core_data.data_point import FlexibleDatasetPoint
from main.core_data.loader import DataPointsLoader
from main.core_data.media.assessment.assessment import Dominance, Valence, Arousal
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

        dreamer = loadmat(f"{self.base_path}/DREAMER.mat")["DREAMER"][0][0]

        # 5 is index of n experiments 4 is index of participants
        self.length = (dreamer[5] * dreamer[4]).item()
        return self.length

    # TODO finish
    def scan(self) -> Iterator[FlexibleDatasetPoint]:
        dreamer = loadmat(f"{self.base_path}/DREAMER.mat")["DREAMER"][0][0]
        # Dreamer has 10 columns. Of these last is useless.
        # dreamer[0] Contains Age, Gender, EEG (index?), Score Valance, Score Arousal Score Dominance
        # [('Age', 'O'), ('Gender', 'O'), ('EEG', 'O'), ('ECG', 'O'), ('ScoreValence', 'O'), ('ScoreArousal', 'O'), ('ScoreDominance', 'O')]
        # dreamer[3] is the order of EEG channels?
        # After dreamer 6 (Compreso) useless

        # Count number of experiments
        users = dreamer[0][0].shape[0]
        for user in range(users):
            # Thank you matlab
            user_experiment_data = dreamer[0][0][user][0]

            # Experiments with index
            for experiment_index in range(len(user_experiment_data["ScoreValence"][0].squeeze())):
                nei = self.dataset_uid_store.uid(str(user), str(experiment_index), "DREAMER")
                # EEG Data: Stimuli is what matters to us. The normalization of bias is not of help? Guess not
                eeg_np = user_experiment_data["EEG"][0][0]["stimuli"][0][experiment_index][0].T  # Time x Channels
                info = mne.create_info(
                    ch_names=self.config.eeg_source_config.get_CH_NAMES(),
                    ch_types=self.config.eeg_source_config.get_CH_TYPES(),
                    sfreq=self.config.eeg_source_config.fs
                )
                raw: RawArray = mne.io.RawArray(eeg_np, info=info, verbose=False)
                valence = user_experiment_data["ScoreValence"][0].squeeze()[experiment_index]
                arousal = user_experiment_data["ScoreArousal"][0].squeeze()[experiment_index]
                dominance = user_experiment_data["ScoreDominance"][0].squeeze()[experiment_index]

                metadata = MetaObject(
                    experiment=nei, dataset_id=self.DATASET_ID, person_id=user, trial=experiment_index
                )

                yield FlexibleDatasetPoint(
                    nei,
                    EEG(eid=nei, data=raw.copy().pick(["eeg"]), fs=raw.info['sfreq']).as_mod_tuple(),
                    # For evaluation
                    Dominance(eid=nei, data=dominance, rating_scale=(1, 5)).as_mod_tuple(),
                    Valence(eid=nei, data=valence, rating_scale=(1, 5)).as_mod_tuple(),
                    Arousal(eid=nei, data=arousal, rating_scale=(1, 5)).as_mod_tuple(),
                    Metadata(data=asdict(metadata), eid=nei).as_mod_tuple()
                )

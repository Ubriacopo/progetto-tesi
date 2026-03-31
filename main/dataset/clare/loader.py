from dataclasses import asdict
from pathlib import Path
from typing import Iterator

import mne
import numpy as np
import pandas as pd
import torch
from mne.io import RawArray

from main.core_data.data_point import FlexibleDatasetPoint
from main.core_data.loader import DataPointsLoader
from main.core_data.media.assessment.assessment import Assessment
from main.core_data.media.ecg import ECG
from main.core_data.media.eeg import EEG
from main.core_data.media.metadata.metadata import MetaObject, Metadata
from main.dataset.clare.config import ClareConfig
from main.dataset.utils import DatasetUidStore


class ClarePointsLoader(DataPointsLoader):
    DATASET_ID: int = 3

    def __init__(self, base_path: str, dataset_uid_store: DatasetUidStore, config: ClareConfig = ClareConfig()):
        super().__init__(dataset_uid_store)
        self.base_path = base_path
        self.config: ClareConfig = config

        self.length: int = 0

    def __len__(self) -> int:
        return 19 * 4 * 55

    def scan(self) -> Iterator[FlexibleDatasetPoint]:
        user_id_list = [x.name for x in Path(self.base_path + "EEG/").iterdir() if x.is_dir()]
        for user_id in user_id_list:
            try:
                # Labels are aggregated together in one file
                labels = pd.read_csv(self.base_path + f"Labels/{user_id}.csv")
                # Other are split per experiment (Everyone did 4 experiments)
                for experiment in range(4):
                    try:
                        eeg_df = pd.read_csv(self.base_path + f"EEG/{user_id}/eeg_data_exp_{experiment}.csv")
                        ecg_df = pd.read_csv(self.base_path + f"ECG/{user_id}/ecg_data_experiment_{experiment}.csv")

                        if eeg_df["Timestamp"].iloc[0] > 1:
                            eeg_df["Timestamp"] = eeg_df["Timestamp"] - eeg_df["Timestamp"].iloc[0]

                        if ecg_df["Timestamp"].iloc[0] > 1:
                            ecg_df["Timestamp"] = ecg_df["Timestamp"] - ecg_df["Timestamp"].iloc[0]

                        current_labels = labels[f"level_{experiment}"].to_list()
                        for sub_experiment, label in enumerate(current_labels):
                            eid = self.dataset_uid_store.uid(str(user_id), str(experiment * 10 + sub_experiment),
                                                             "CLARE")

                            # Each experiment has range 10 seconds
                            eeg = eeg_df[(eeg_df["Timestamp"] >= sub_experiment * 10) &
                                         (eeg_df["Timestamp"] <= (sub_experiment + 1) * 10)]
                            eeg = eeg.to_dict(orient="list")
                            eeg.pop("Timestamp")
                            eeg = self.config.eeg_source_config.channels_remap(eeg)

                            info = mne.create_info(
                                ch_names=self.config.eeg_source_config.get_CH_NAMES(),
                                ch_types=self.config.eeg_source_config.get_CH_TYPES(),
                                sfreq=self.config.eeg_source_config.fs
                            )

                            raw: RawArray = mne.io.RawArray(np.array(list(eeg.values())), info=info, verbose=False)

                            ecg = ecg_df[(ecg_df["Timestamp"] >= sub_experiment * 10) &
                                         (ecg_df["Timestamp"] <= (sub_experiment + 1) * 10)]

                            ll_ra = ecg["ECG LL-RA CAL"].to_numpy()
                            la_ra = ecg["ECG LA-RA CAL"].to_numpy()
                            vx_rl = ecg["ECG Vx-RL CAL"].to_numpy()  # optional

                            ecg_tensor = np.stack([ll_ra, la_ra, vx_rl], axis=0)  # [C, T]
                            ecg_tensor = torch.from_numpy(ecg_tensor).float()

                            metadata = MetaObject(experiment=eid, dataset_id=self.DATASET_ID, person_id=user_id,
                                                  trial=experiment * 10 + sub_experiment, )

                            yield FlexibleDatasetPoint(
                                eid,
                                EEG(eid=eid, data=raw.copy().pick(["eeg"]), fs=raw.info["sfreq"]).as_mod_tuple(),
                                ECG(eid=eid, data=ecg_tensor, leads=self.config.ecg_source_config.LEAD_NAMES,
                                    fs=512).as_mod_tuple(),
                                Assessment(eid=eid, data=label, labels=self.config.score_labels_config.labels,
                                           scales=self.config.score_labels_config.scales).as_mod_tuple(),
                                Metadata(data=asdict(metadata), eid=eid).as_mod_tuple()
                            )

                    except Exception as e:
                        self.logger.error(f"Loading failed for {experiment} of {user_id}. Procedure will continue and drop the element")
                        self.logger.error(e)

            except Exception as e:
                self.logger.error(f"Loading failed for {user_id}. Procedure will continue and drop the element")
                self.logger.error(e)

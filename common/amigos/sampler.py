import dataclasses
import gc
import re
from abc import ABC, abstractmethod
from pathlib import Path

import ffmpeg
import numpy as np
import pandas as pd

from common.amigos.data_sampler import DataSampler, DatasetDataCollection, extract_trial_data, load_participant_data
from utils.data import pad_main_axis


@dataclasses.dataclass
class DatasetDataCollection:
    experiment_id: str

    eeg_data: np.ndarray | list[np.ndarray]
    mediafile_path: str | Path
    frontal_media_path: str


@dataclasses.dataclass
class SamplingResult:
    # The two relevant infos
    media_path: str
    data: np.ndarray

    # Metadata
    original_filename: str
    split_index: int

    start_timestamp: int
    stop_timestamp: int

    experiment_id: str


# todo controlla
# Questa classe legge i file e li conserva come experiments.
class AMIGOSCollector:
    def __init__(self, base_path: str):
        self.base_path: str = base_path
        self.data_collection: list[DatasetDataCollection] = []

        self.scanned: bool = False

    def scan(self, force: bool = False) -> list[DatasetDataCollection]:
        if self.scanned and not force:
            return self.data_collection  # Scan is not performed again

        processed_data = Path(self.base_path + "pre_processed_py/")

        if not processed_data.exists():
            for f in Path(self.base_path + "pre_processed").iterdir():
                extract_trial_data(self.base_path + "pre_processed_py/", str(f))

        participant_data = load_participant_data(Path(self.base_path + "pre_processed_py/"))
        face_video_folder = self.base_path + "face/"
        face_folder = Path(face_video_folder)

        for v in face_folder.iterdir():
            # [0] -> P40 [1] -> 18 [2] -> face(.mov) (Stemmed)
            person, video_id, _ = v.stem.split("_")
            # Add missing prefix zero to match the np data
            person = re.sub(r'([A-Z])(\d)\b', r'\g<1>0\2', person)

            experiment_id = person + "_" + video_id

            video_index = np.where(participant_data[person]["VideoIDs"] == video_id)[0]
            eeg_data = participant_data[person]["joined_data"][video_index]
            self.data_collection.append(DatasetDataCollection(experiment_id, eeg_data[0], str(v), ""))

        self.scanned = True
        return self.data_collection


class AMIGOSSampler(DataSampler, ABC):
    # The given settings are the default set
    def __init__(self, output_path: str, max_duration: int, fps: int = 15, audio_rate: int = 24000, data_fs: int = 128):
        self.output_path: str = output_path
        self.max_duration: int = max_duration
        self.fps: int = fps
        self.audio_rate: int = audio_rate
        self.fs: int = data_fs

    def process_media(self, sample: DatasetDataCollection, segments: list[tuple[int, int]]):
        for idx, (start, end) in enumerate(segments):
            # How long the clip takes
            length = end - start
            Path(self.output_path + f'{sample.experiment_id}').mkdir(parents=True, exist_ok=True)
            output_path = self.output_path + f'{sample.experiment_id}/{idx}.mp4'

            p = ffmpeg.input(sample.mediafile_path, ss=start, t=length, r=self.fps)
            p.output(output_path, c='copy', ar=self.audio_rate).run()

            yield output_path

    def process_data(self, sample: DatasetDataCollection, segments: list[tuple[int, int]]):
        for start, stop in segments:
            eeg = sample.eeg_data[start * self.fs:stop * self.fs]
            pad_width = self.fs * self.max_duration - eeg.shape[0]
            yield pad_main_axis(np.array(eeg), pad_width)

    def process_sample(self, sample: DatasetDataCollection):
        # How long the media lasts
        segments = self.compute_segments(sample)

        # 0s interval might happen on last one? TODO: Sarebbe compito di implementation
        final_start, final_stop = segments[-1]
        if final_start == final_stop: segments.pop()
        generator = zip(self.process_media(sample, segments), self.process_data(sample, segments))

        for idx, (media, eeg) in enumerate(generator):
            start, stop = segments[idx]
            yield SamplingResult(media, eeg, sample.mediafile_path, idx, start, stop, sample.experiment_id)

    @abstractmethod
    def compute_segments(self, sample: DatasetDataCollection) -> list[tuple[int, int]]:
        pass


class SimpleAMIGOSSampler(AMIGOSSampler):
    def compute_segments(self, sample: DatasetDataCollection) -> list[tuple[int, int]]:
        # From EEG Signal we can understand how long the video really is. (As we know Hz)
        duration = sample.eeg_data.shape[0] / self.fs

        starts = np.arange(0, duration, self.max_duration).astype(int)
        stops = np.minimum(starts + self.max_duration, duration).astype(int)

        return list(zip(starts, stops))


def __store_df(data_filepath: str, csv_filepath: str, df: pd.DataFrame):
    np.save(data_filepath, df["data"].to_numpy())
    df = df.drop("data", axis=1)
    df["data_file"] = data_filepath

    df.to_csv(csv_filepath, index=True)

    # To avoid having too much data up the RAM
    del df
    gc.collect()


def do_sample(base_path: str, entries_per_file: int = 200):
    files = AMIGOSCollector(base_path).scan()
    sampler = SimpleAMIGOSSampler(base_path + "sampled/", 5)

    generated_file_number = 0

    df: pd.DataFrame | None = None
    npy: np.ndarray | None = None

    for entry in files:
        for res in sampler.process_sample(entry):
            res_df = pd.DataFrame([res])
            # Extract the data and keep it safe
            npy = np.append(npy, res_df["data"].to_numpy()) if npy is not None else res_df["data"].to_numpy()
            res_df = res_df.drop("data", axis=1)

            res_df["data_file"] = base_path + "sampled/AMIGOS_split_" + str(generated_file_number)
            res_df["data_index"] = len(npy) - 1  # We store the index of the npy file

            df = pd.DataFrame(res_df) if df is None else pd.concat([df, res_df], ignore_index=True)

            if len(df) % entries_per_file == 0:
                file_template = base_path + "sampled/AMIGOS_split_" + str(generated_file_number)
                np.save(file_template + ".npy", npy)

                # Reset state of npy variable
                del npy
                npy = None

                generated_file_number += 1

    file_template = base_path + "sampled/AMIGOS_split_" + str(generated_file_number)
    np.save(file_template + ".npy", npy)

    df.to_csv(base_path + "sampled/AMIGOS_sampled" + ".csv", index=True)

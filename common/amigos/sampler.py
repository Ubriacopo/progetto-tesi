import dataclasses
import re
import traceback
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from itertools import repeat
from pathlib import Path

import ffmpeg
import numpy as np
import pandas as pd
import speech_recognition as sr
from moviepy import VideoFileClip

from common.amigos.utils import load_participant_data, extract_trial_data
from common.ds.sampler import DataSampler
from utils.data import pad_main_axis


@dataclasses.dataclass
class DatasetDataCollection:
    experiment_id: str
    eeg_data: list[np.ndarray] | np.ndarray
    mediafile_path: str | Path  # todo mp3 e mp4 cambiano ora struttua
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


@dataclasses.dataclass
class StoredSamplingResult(SamplingResult):
    data_file: str
    data_index: int


# Questa classe legge i file e li conserva come experiments.
class AMIGOSCollector:
    def __init__(self, base_path: str):
        """
        todo describe
        :param base_path:
        """
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
            self.data_collection.append(DatasetDataCollection(experiment_id, eeg_data[0], str(v.resolve()), ""))

        self.scanned = True
        return self.data_collection


class AMIGOSSampler(DataSampler, ABC):
    def __init__(self, output_path: str, max_duration: int, fps: int = 15, audio_rate: int = 24000, data_fs: int = 128):
        self.output_path: str = output_path
        self.max_duration: int = max_duration
        self.fps: int = fps
        self.audio_rate: int = audio_rate
        self.fs: int = data_fs

    def get_generated_media_extensions(self) -> tuple[str, str]:
        return '.mp4', '.wav'

    @abstractmethod
    def process_split_media(self, idx: int, times: tuple, sample: DatasetDataCollection) -> str:
        pass

    def process_split_data(self, times: tuple, sample: DatasetDataCollection):
        start, end = times

        eeg = sample.eeg_data[start * self.fs:end * self.fs]
        pad_width = self.fs * self.max_duration - eeg.shape[0]
        return pad_main_axis(np.array(eeg), pad_width)

    @abstractmethod
    def compute_segments(self, sample: DatasetDataCollection) -> list[tuple[int, int]]:
        pass

    def _process_split(self, sample: DatasetDataCollection, index: int, segment: tuple[int, int]):
        start, stop = segment
        # When processing a split we have to handle both media + data
        data = self.process_split_data(segment, sample)
        media = self.process_split_media(index, segment, sample)
        return SamplingResult(media, data, sample.mediafile_path, index, start, stop, sample.experiment_id)

    def process_sample(self, sample: DatasetDataCollection, concurrent: bool = True):
        # How long the media lasts
        segments = self.compute_segments(sample)
        # 0s interval might happen on last one? TODO: Sarebbe compito di implementation
        final_start, final_stop = segments[-1]
        if final_start == final_stop:
            segments.pop()

        if concurrent:
            with ThreadPoolExecutor() as executor:
                results = executor.map(self._process_split, repeat(sample), range(len(segments)), segments)
                for res in results: yield res  # Yield
        else:
            for idx, segment in enumerate(segments):
                yield self._process_split(sample, idx, segment)


class FastTextlessAMIGOSampler(AMIGOSSampler, ABC):
    def process_split_media(self, idx: int, times: tuple, sample: DatasetDataCollection) -> str:
        try:
            start, end = times
            output_path = self.output_path + f'{sample.experiment_id}/{idx}'

            length = end - start
            Path(self.output_path + f'{sample.experiment_id}').mkdir(parents=True, exist_ok=True)
            p = ffmpeg.input(sample.mediafile_path, ss=start, t=length)
            p.output(output_path + ".mp4", map='0:v', r=self.fps, loglevel="quiet").run()
            p = ffmpeg.input(sample.mediafile_path, ss=start, t=length)
            p.output(output_path + ".wav", acodec='pcm_s16le', map='0:a', ar=self.audio_rate, loglevel="quiet").run()

            return str(Path(output_path).resolve())

        except Exception as exception:
            traceback.print_exc()
            print(exception)
            return " "


class MoviepyAMIGOSSampler(AMIGOSSampler, ABC):

    def __init__(self, output_path: str, max_duration: int, fps: int = 15, audio_rate: int = 24000, data_fs: int = 128,
                 process_text: bool = True):
        super().__init__(output_path, max_duration, fps, audio_rate, data_fs)
        self.process_text = process_text

    def process_split_media(self, idx: int, times: tuple, sample: DatasetDataCollection) -> str:
        try:
            video = VideoFileClip(sample.mediafile_path)
            start, end = times

            output_path = self.output_path + f'{sample.experiment_id}/{idx}'
            Path(self.output_path + f'{sample.experiment_id}').mkdir(parents=True, exist_ok=True)

            clip = video.subclipped(start, end)
            clip.with_fps(self.fps).write_videofile(output_path + ".mp4")
            clip.audio.with_fps(self.audio_rate).write_audiofile(output_path + ".wav", codec="pcm_s16le")

            if self.process_text:
                self.extract_text(output_path)

            return str(Path(output_path).resolve())

        except Exception as e:
            traceback.print_exc()
            print(e)
            return " "

    def extract_text(self, output_path: str):
        recognizer = sr.Recognizer()
        with sr.AudioFile(output_path + ".wav") as source:
            audio_data = recognizer.record(source)

            try:
                # Sphinx is bad as stated by some sources
                # Locally hosted. Might opt for stronger models later todo check better alternatives
                # todo clean better the audio?
                text = recognizer.recognize_sphinx(audio_data)
                # todo: boh crasha
                # text = recognizer.recognize_faster_whisper(audio_data, model="base")
            except Exception as e:
                print("Exception met when handling the speech recognition")
                text = ""

            with open(output_path + ".txt", "w") as file:
                file.write(f"{text}\n")


class TimedSegmentsAMIGOSSampler(AMIGOSSampler, ABC):
    def compute_segments(self, sample: DatasetDataCollection) -> list[tuple[int, int]]:
        # From EEG Signal we can understand how long the video really is. (As we know Hz)
        duration = sample.eeg_data.shape[0] / self.fs

        starts = np.arange(0, duration, self.max_duration).astype(int)
        stops = np.minimum(starts + self.max_duration, duration).astype(int)

        return list(zip(starts, stops))


class TimedMoviepyAMIGOSSampler(TimedSegmentsAMIGOSSampler, MoviepyAMIGOSSampler):
    pass


class TimedFastAMIGOSSampler(TimedSegmentsAMIGOSSampler, FastTextlessAMIGOSampler):
    pass


def do_sample(base_path: str, output_path: str, sampler: AMIGOSSampler, entries_per_file: int = 200, concurrent=False):
    files = AMIGOSCollector(base_path).scan()
    generated_file_number = 0

    df: pd.DataFrame | None = None
    npy: np.ndarray | None = None

    for entry in files:
        for res in sampler.process_sample(entry, concurrent):
            res_df = pd.DataFrame([res])
            # Extract the data and keep it safe
            npy = np.append(npy, res_df["data"].to_numpy()) if npy is not None else res_df["data"].to_numpy()
            res_df = res_df.drop("data", axis=1)

            file_template = output_path + "AMIGOS_split_" + str(generated_file_number) + ".npy"
            res_df["data_file"] = Path(file_template).resolve()
            res_df["data_index"] = len(npy) - 1  # We store the index of the npy file

            df = pd.DataFrame(res_df) if df is None else pd.concat([df, res_df], ignore_index=True)

            if len(df) % entries_per_file == 0:
                np.save(file_template, npy)
                # Reset state of npy variable
                del npy
                npy = None
                generated_file_number += 1

    # Store the descriptor file of this dataset.
    file_template = output_path + "AMIGOS_split_" + str(generated_file_number) + ".npy"
    np.save(file_template, npy)

    df.to_csv(output_path + "AMIGOS_sampled" + ".csv", index=True)

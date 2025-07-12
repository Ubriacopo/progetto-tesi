import gc
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Generator

import cv2
import ffmpeg
import numpy as np
from moviepy import VideoFileClip
from scipy.io import loadmat

from common.data.video import extract_frames
from utils.data import pad_main_axis


@dataclass
class DatasetDataCollection:
    experiment_id: str

    eeg_data: np.ndarray | list[np.ndarray]
    mediafile_path: str | Path
    frontal_media_path: str


def extract_trial_data(destination_path: str, source_path: str):
    """
    The data is converted to a numpy friendly type to help us work better (we have some advantages).
    Consideration:
        Load all EEG (40 × 100MB = ~4GB) into RAM at startup — easy with 128GB. (Our server)

    :param destination_path:
    :param source_path:
    """
    mat = loadmat(source_path)  # Source file

    data = {k: v for k, v in mat.items() if not k.startswith("__")}
    for key in data:
        # Remove the heading dimension
        data[key] = data[key].squeeze()

    file_name = Path(source_path).stem

    output_path = Path(f"{destination_path}/{file_name}.npz")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **data)


def load_participant_data(data_path: Path) -> dict:
    return_object = {}
    for f in data_path.iterdir():
        if not f.is_file() or f.suffix != '.npz':
            continue

        # Suffix like _P01.mat so take it stemmed
        return_object[f.stem.split("_")[-1]] = np.load(f, allow_pickle=True)

    return return_object


class DataSampler(ABC):
    @abstractmethod
    def process_sample(self, sample):
        pass


# Questa classe legge i file e li conserva come experiments.
class AMIGOSCollector:
    def __init__(self, base_path: str):
        self.base_path: str = base_path
        self.data_collection: list[DatasetDataCollection] = []

        self.scanned: bool = False

    def scan(self, force: bool = False):
        if self.scanned and not force:
            return  # Scan is not performed again

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


class AMIGOSSampler(DataSampler):
    def __init__(self, video_out_path: str, fps: int = 15, max_segment_seconds: int = 5):
        self.fps, self.max_segment_seconds = fps, max_segment_seconds
        self.video_out_path = video_out_path
        self.audio_hz = 24000
        self.fs = 128

    def process_video(self, sample: DatasetDataCollection, segments: list[tuple[int, int]]):
        # Downsample
        for i, (start, end) in enumerate(segments):
            length = end - start
            Path(self.video_out_path + f'{sample.experiment_id}').mkdir(parents=True, exist_ok=True)
            output_path = self.video_out_path + f'{sample.experiment_id}/{i}.mp4'
            ffmpeg.input(sample.mediafile_path, ss=start, t=length, r=self.fps).output(output_path, c='copy').run()
            yield output_path

    def process_audio(self, sample: DatasetDataCollection, segments: list[tuple[int, int]]):
        original_video = VideoFileClip(sample.mediafile_path)
        for start, stop in segments:
            clip: VideoFileClip = original_video.subclipped(start, stop)
            audio = clip.audio.to_soundarray(fps=self.audio_hz)
            pad_width = self.audio_hz * self.max_segment_seconds - audio.shape[0]
            yield pad_main_axis(np.array(audio), pad_width)

    def process_eeg(self, sample: DatasetDataCollection, segments: list[tuple[int, int]]) -> Generator:
        for start, stop in segments:
            eeg = sample.eeg_data[start * self.fs:stop * self.fs]
            pad_width = self.fs * self.max_segment_seconds - eeg.shape[0]
            yield pad_main_axis(np.array(eeg), pad_width)

    def process_sample(self, sample: DatasetDataCollection):
        # todo: Generate segments from data
        duration = sample.eeg_data.shape[0] / self.fs  # Downsampled to 128Hz

        # Questa tecnica è la banale divisione per intervalli regolari.
        # TODO: dividere su euristica di picchi con piccolo overlap
        starts = np.arange(0, duration, self.max_segment_seconds).astype(int)
        stops = np.minimum(starts + self.max_segment_seconds, duration).astype(int)

        segments = list(zip(starts, stops))
        final_start, final_stop = segments[-1]

        if final_start == final_stop:
            segments.pop()

        video_generator = self.process_video(sample, segments)
        # audio_generator = self.process_audio(sample, segments)
        eeg_generator = self.process_eeg(sample, segments)

        # todo: Text generator

        index = 0

        print("Working on file:" + sample.mediafile_path)
        for video, eeg in zip(video_generator, eeg_generator):
            yield {
                "video": video, "text": "", "eeg_data": eeg,
                "metadata": {"original_video": sample.mediafile_path, "split": index, "timestamps": segments[index]}
            }

            index += 1


def sample_amigos(base_path: str, entries_per_file: int = 35):
    collector = AMIGOSCollector(base_path)
    collector.scan()
    # todo log a little bit
    sampler = AMIGOSSampler(base_path + "sampled/")

    # todo pad to 5s?
    video_list, text_list, eeg_list, timestamps, filenames = [], [], [], [], []
    generated_file_number = 0

    Path(base_path + "/compressed/").mkdir(parents=True, exist_ok=True)
    for x in collector.data_collection:
        segments = sampler.process_sample(x)
        for y in segments:
            v, t, eeg, meta = y.values()

            video_list.append(v)
            text_list.append(t)
            eeg_list.append(eeg)

            # TODO Se non faccio padding non puo funzionare per shape di array

            timestamps.append(meta["timestamps"])
            filenames.append(meta["original_video"])
            print(f"Video:{len(video_list)}")
            if len(video_list) % entries_per_file == 0:
                # store the file
                # todo dataclass
                save_data = {
                    "video": video_list, "text": text_list, "eeg_data": eeg_list,
                    "timestamp": timestamps, "filename": filenames
                }
                # TODO: Al posto di salvare tutto compresso come npz andiamo a creare un file video downasmpled.
                #       Era carino fare tutto in unico file ma melgio fare clip piccole e salvarle
                output_path = base_path + "/compressed/" + "AMIGOS_compressed_" + str(generated_file_number) + ".npz"
                np.savez_compressed(output_path, **save_data)
                generated_file_number += 1

                # Reset
                # todo gc collection right
                del save_data
                del video_list, text_list, eeg_list, timestamps, filenames
                video_list, text_list, eeg_list, timestamps, filenames = [], [], [], [], []

                gc.collect()

    save_data = {
        "video": video_list, "text": text_list, "eeg_data": eeg_list,
        "timestamp": timestamps, "filename": filenames
    }
    output_path = base_path + "/compressed/" + "AMIGOS_compressed_" + str(generated_file_number) + ".npz"
    np.savez(output_path, data=np.array(save_data, dtype=object))
    del video_list, audio_list, text_list, eeg_list, timestamps, filenames
    video_list, audio_list, text_list, eeg_list, timestamps, filenames = [], [], [], [], [], []

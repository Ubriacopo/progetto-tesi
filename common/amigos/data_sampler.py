import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

import cv2
import ffmpeg
import numpy as np
from moviepy import VideoFileClip
from scipy.io import loadmat

from common.data.video import downsample_video, extract_frames


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
            self.data_collection.append(DatasetDataCollection(experiment_id, eeg_data, str(v), ""))

        self.scanned = True


class AMIGOSSampler(DataSampler):
    def __init__(self, fps: int = 15, max_segment_seconds: int = 5):
        self.fps, self.max_segment_seconds = fps, max_segment_seconds
        self.audio_hz = 16000

    def process_video(self, sample: DatasetDataCollection, segments: list[tuple[int, int]]):
        # Downsample
        temp_file_path = Path(sample.mediafile_path + ".temp.mov")
        ffmpeg.input(sample.mediafile_path).output(str(temp_file_path), r=self.fps).run()

        # Read the frames
        video = cv2.VideoCapture(str(temp_file_path))
        frames = extract_frames(video)

        # Split in small intervals
        for start, stop in segments:
            yield frames[int(start * self.fps):int(stop * self.fps)]

        temp_file_path.unlink()

    def process_audio(self, sample: DatasetDataCollection, segments: list[tuple[int, int]]):
        original_video = VideoFileClip(sample.mediafile_path)
        for start, stop in segments:
            clip: VideoFileClip = original_video.subclipped(start, stop)
            yield clip.audio.to_soundarray(self.audio_hz)

    def process_sample(self, sample: DatasetDataCollection):
        # todo: Generate segments from data
        duration = sample.eeg_data[0].shape[0] / 128  # Downsampled to 128Hz

        # Questa tecnica è la banale divisione per intervalli regolari.
        # TODO: dividere su euristica di picchi con piccolo overlap
        starts = np.arange(0, duration, self.max_segment_seconds)
        stops = np.minimum(starts + self.max_segment_seconds, duration).astype(int)

        segments = list(zip(starts, stops))
        final_start, final_stop = segments[-1]

        if final_start == final_stop:
            segments.pop()

        # todo sample.eeg_data va diviso anche questo negli intervalli del video

        video_generator = self.process_video(sample, segments)
        audio_generator = self.process_audio(sample, segments)
        # todo: Text generator

        for video, audio in zip(video_generator, audio_generator):
            yield {"video": video, "audio": audio, "text": "", "eeg_data": sample.eeg_data}


def sample_amigos(base_path: str, entries_per_file: int = 150):
    collector = AMIGOSCollector(base_path)
    collector.scan()
    # todo log a little bit
    sampler = AMIGOSSampler()
    video_list, audio_list, text_list, eeg_list = [], [], [], []
    generated_file_number = 0

    Path(base_path + "/compressed/").mkdir(parents=True, exist_ok=True)
    for x in collector.data_collection:
        segments = sampler.process_sample(x)
        for y in segments:
            v, a, t, eeg = y.values()

            video_list.append(v)
            audio_list.append(a)
            text_list.append(t)
            eeg_list.append(eeg)

            if len(video_list) % entries_per_file == 0:
                # store the file
                save_data = {"video": video_list, "audio": audio_list, "text": text_list, "eeg_data": eeg_list}
                output_path = base_path + "/compressed/" + "AMIGOS_compressed_" + str(generated_file_number) + ".npz"
                np.savez(output_path, **save_data)
                generated_file_number += 1

    save_data = {"video": video_list, "audio": audio_list, "text": text_list, "eeg_data": eeg_list}
    output_path = base_path + "/compressed/" + "AMIGOS_compressed_" + str(generated_file_number) + ".npz"
    np.savez(output_path, data=np.array(save_data, dtype=object))

import traceback
from abc import abstractmethod, ABC
from dataclasses import replace
from pathlib import Path
from typing import Optional

import pandas as pd

from common.data.eeg import EEGToMneRawFromChannels
from common.data.loader import DatasetDataPoint, DataLoader, EEGDatasetDataPoint
from common.data.sampler import Segmenter
from common.data.transform import Compose

SPEC_FILE_NAME: str = "spec.csv"


class Preprocessor(ABC):
    def __init__(self, output_path: str):
        self.output_path: str = output_path

    @abstractmethod
    def preprocess(self, x: DatasetDataPoint) -> DatasetDataPoint | list[DatasetDataPoint]:
        pass

    def run(self, loader: DataLoader) -> bool:
        try:
            # Read an existing spec if it was computed.
            existing_df: Optional[pd.DataFrame] = None
            existing_path = self.output_path + SPEC_FILE_NAME
            if Path(existing_path).exists():
                existing_df = pd.read_csv(existing_path)

            docs: list[DatasetDataPoint] = []
            # todo if multitreading do it here on samples
            for i in loader.scan():
                key = i.get_identifier()
                if existing_df is not None and existing_df[key].str.contains(i.entry_id).any():
                    continue  # This element was already processed.

                [docs.append(e) for e in self.preprocess(i)]
                df = pd.DataFrame([d.to_dict() for d in docs])
                if existing_df is not None:
                    df = pd.concat([df, existing_df], ignore_index=True)

                df.to_csv(self.output_path + "spec.csv", index=False)
                existing_df = df

            print("Procedure finished correctly.")
            print("Spec file can found at:", self.output_path, "spec.csv")
            return True

        except Exception as e:
            print("Preprocessing pipeline failed for an unexpected error:", e)
            traceback.print_exc()
            return False


class EEGSegmenterPreprocessor(Preprocessor):
    def __init__(self, output_path: str, segmenter: Segmenter,
                 # In order to work with EEG data
                 ch_names: list[str], ch_types: list[str],
                 # Custom pipelines to insert (Composition over inheritance).
                 sample_pipeline: Compose = None, split_pipeline: Compose = None):
        super().__init__(output_path)

        self.segmenter: Segmenter = segmenter
        self.sample_pipeline: Optional[Compose] = sample_pipeline
        self.split_pipeline: Optional[Compose] = split_pipeline

        # EEG mapping for mne
        self.ch_names: list[str] = ch_names
        self.ch_types: list[str] = ch_types

    def preprocess(self, x: EEGDatasetDataPoint) -> list[DatasetDataPoint]:
        # Process output from x (in samples) will be stored under /{output_path]/{x._entry_id}+SUFFIX
        original_sample_id = x.entry_id

        if x.eeg.data.shape[0] != len(self.ch_names):
            x.eeg.data = x.eeg.data.T  # Transpose

        assert x.eeg.data.shape[0] == len(self.ch_names), "Shape mismatch for EEG data"
        x = EEGToMneRawFromChannels(channel_names=self.ch_names, channel_types=self.ch_types)(x)
        if self.sample_pipeline is not None:
            x = self.sample_pipeline(x)

        segments: list[tuple[int, int]] = self.segmenter.compute_segments(x)
        x_out_folder = self.output_path + x.entry_id + "/"
        Path(x_out_folder).mkdir(parents=True, exist_ok=True)
        x_segments = [self.preprocess_segment(x, idx, segment, x_out_folder) for idx, segment in enumerate(segments)]

        eeg_out_path: str = self.output_path + f'{original_sample_id}.fif'

        x.eeg.data.save(eeg_out_path, overwrite=True, split_size="2GB")
        for x_segment in x_segments:
            x_segment.eeg.file_path = eeg_out_path

        return x_segments

    def preprocess_segment(self, x: EEGDatasetDataPoint, idx: int,
                           segment: tuple[int | float, int | float], out_folder: str) -> EEGDatasetDataPoint:
        nid = x.entry_id + "_" + str(idx)
        y = EEGDatasetDataPoint(
            entry_id=nid,
            # No files path are associated now.
            eeg=replace(x.eeg, interval=segment),
            vid=replace(x.vid, interval=segment),
            aud=replace(x.aud, interval=segment),
            txt=replace(x.txt) if x.txt is not None else None,
        )

        if self.split_pipeline is not None:
            y: EEGDatasetDataPoint = self.split_pipeline(y)

        # Save Video
        vid_out_path = out_folder + f'{y.entry_id}.mp4'
        y.vid.data.write_videofile(vid_out_path, audio=False, codec="libx264", ffmpeg_params=["-pix_fmt", "yuv420p"], )
        y.vid.file_path = vid_out_path

        # Save Audio
        aud_out_path = out_folder + f'{y.entry_id}.wav'
        y.aud.data.write_audiofile(aud_out_path)
        y.aud.file_path = aud_out_path

        # Save Text if exists
        if y.txt is not None and y.txt.data is not None:
            txt_out_path = out_folder + f'{y.entry_id}.txt'
            with open(txt_out_path, "w", encoding="utf-8") as f:
                f.write(y.txt.data)

        # EEG data is treated differently and aggregated to save space.
        # So it is not performed here.
        return y

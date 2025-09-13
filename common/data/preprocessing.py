import os.path
import traceback
from abc import abstractmethod, ABC
from dataclasses import replace
from pathlib import Path
from typing import Optional, Text, TypeVar, Generic

import numpy as np
import pandas as pd
import torch

from common.data.audio import Audio
from common.data.data_point import DatasetDataPoint, EEGDatasetDataPoint, EEGDatasetTransformWrapper, call_pipelines, \
    AgnosticDatasetPoint, AgnosticDatasetTransformWrapper
from common.data.eeg import EEG
from common.data.eeg.transforms import EEGToMneRaw
from common.data.loader import DataPointsLoader
from common.data.media import sanitize_for_ast
from common.data.sampler import Segmenter
from common.data.utils import build_tensor_dict
from common.data.video import Video

SPEC_FILE_NAME: str = "spec.csv"

T = TypeVar("T")


class Preprocessor(ABC, Generic[T]):
    def __init__(self, output_path: str):
        """
        Creates a processed dataset in a target folder. Info of the new ds are contained in the spec.csv
        """
        self.output_path: str = output_path

    @abstractmethod
    def preprocess(self, x: T) -> dict | list[dict]:
        pass

    @abstractmethod
    def export(self, x: list[T], output_path: str) -> None:
        pass

    def run(self, loader: DataPointsLoader) -> bool:
        try:
            # Read an existing spec if it was computed.
            existing_df: Optional[pd.DataFrame] = None
            existing_path = self.output_path + SPEC_FILE_NAME
            Path(self.output_path).mkdir(parents=True, exist_ok=True)
            if Path(existing_path).exists():
                existing_df = pd.read_csv(existing_path)

            docs: list[dict] = []
            # todo: If multithreading do it here on samples. Consigliano Queue per generare objects
            for i in loader.scan():
                key = i.get_identifier()
                if existing_df is not None and existing_df[key].str.contains(i.eid).any():
                    continue  # This element was already processed.

                [docs.append(e) for e in self.preprocess(i)]
                df = pd.DataFrame([d for d in docs])
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


class TorchExportsSegmenterPreprocessor(Preprocessor[AgnosticDatasetPoint]):
    def __init__(self, output_path: str, segmenter: Segmenter,
                 # In order to work with EEG data
                 ch_names: list[str], ch_types: list[str], pipeline: AgnosticDatasetTransformWrapper):
        super().__init__(output_path)
        self.segmenter: Segmenter = segmenter
        self.pipeline: AgnosticDatasetTransformWrapper = pipeline
        # EEG mapping for mne
        self.ch_names: list[str] = ch_names
        self.ch_types: list[str] = ch_types

    def preprocess(self, x: AgnosticDatasetPoint) -> dict | list[dict]:
        if not hasattr(x, EEG.modality_code()):
            raise ValueError("EEG data is required by design in any dataset")

        if x[EEG.modality_code()].data.shape[0] != len(self.ch_names):
            x[EEG.modality_code()].data = x[EEG.modality_code()].data.T  # Transpose
        assert x[EEG.modality_code()].data.shape[0] == len(self.ch_names), "Shape mismatch for EEG data"
        segments: list[tuple[int, int]] = self.segmenter.compute_segments(x[EEG.modality_code()])

        x_segments = [self.preprocess_segment(x, segment) for idx, segment in enumerate(segments)]

        output_path: str = self.output_path + f'{x.eid}'
        self.export(x_segments, output_path)
        # Return file specification
        return_segments = [
            {"index": idx, x.get_identifier(): x.eid, "segment": segment}
            for idx, (x, segment) in enumerate(zip(x_segments, segments))
        ]
        return_segments = sanitize_for_ast(return_segments)
        return return_segments

    def preprocess_segment(self, x: AgnosticDatasetPoint,
                           segment: tuple[int | float | np.ndarray, int | float | np.ndarray]) -> AgnosticDatasetPoint:
        if isinstance(segment[0], np.ndarray):
            segment = (segment[0].item(), segment[1].item())

        y = x.clone(x.eid)  # entry_id is useless for this approach
        for arg, value in y.__dict__.items():
            if hasattr(value, "interval"):
                value.__setattr__("interval", segment)

        if self.pipeline is None:
            raise ValueError("pipeline is required for preprocessing")
        y = self.pipeline.call(y)
        return y

    def export(self, segments: list[AgnosticDatasetPoint], output_path: str):
        objects = [s.to_dict() for s in segments]
        torch.save(build_tensor_dict(objects), output_path + ".pt")


class EEGSegmenterPreprocessor(Preprocessor):
    def export(self, x: list[DatasetDataPoint | dict], output_path: str):
        pass

    def __init__(self, output_path: str, segmenter: Segmenter,
                 # In order to work with EEG data
                 ch_names: list[str], ch_types: list[str],
                 # Custom pipelines to insert (Composition over inheritance).
                 sample_pipeline: EEGDatasetTransformWrapper = None, split_pipeline: EEGDatasetTransformWrapper = None):
        super().__init__(output_path)

        self.segmenter: Segmenter = segmenter
        # todo AgnosticDatasetTransformWrapper
        self.sample_pipeline: EEGDatasetTransformWrapper = sample_pipeline
        self.split_pipeline: EEGDatasetTransformWrapper = split_pipeline

        # EEG mapping for mne
        self.ch_names: list[str] = ch_names
        self.ch_types: list[str] = ch_types

    def preprocess(self, x: EEGDatasetDataPoint) -> list[dict]:
        # Process output from x (in samples) will be stored under /{output_path]/{x._entry_id}+SUFFIX
        original_sample_id = x.entry_id

        if x.eeg.data.shape[0] != len(self.ch_names):
            x.eeg.data = x.eeg.data.T  # Transpose

        assert x.eeg.data.shape[0] == len(self.ch_names), "Shape mismatch for EEG data"
        x.eeg = EEGToMneRaw(channel_names=self.ch_names, channel_types=self.ch_types)(x.eeg)
        if self.sample_pipeline is not None:
            x = call_pipelines(x, self.sample_pipeline)

        segments: list[tuple[int, int]] = self.segmenter.compute_segments(x)
        x_out_folder = self.output_path + x.entry_id + "/"
        Path(x_out_folder).mkdir(parents=True, exist_ok=True)
        x_segments = [self.preprocess_segment(x, idx, segment, x_out_folder) for idx, segment in enumerate(segments)]

        eeg_out_path: str = self.output_path + f'{original_sample_id}_raw.fif'

        assert (
                isinstance(x.eeg, EEG)
                and isinstance(x.vid, Video) or x.vid is None
                and isinstance(x.txt, Text) or x.txt is None
                and isinstance(x.aud, Audio) or x.aud is None
        ), "EEGDatasetDataPoint are required to not be in tensor form here."

        x.eeg.data.save(eeg_out_path, overwrite=True, split_size="2GB")
        for x_segment in x_segments:
            x_segment.eeg.file_path = os.path.relpath(Path(eeg_out_path).resolve(), self.output_path)

        return [x.to_dict() for x in x_segments]

    def preprocess_segment(self, x: EEGDatasetDataPoint, idx: int,
                           segment: tuple[int | float | np.ndarray, int | float | np.ndarray], out_folder: str) \
            -> EEGDatasetDataPoint:
        if isinstance(segment[0], np.ndarray):
            segment = (segment[0].item(), segment[1].item())

        assert (
                isinstance(x.eeg, EEG)
                and isinstance(x.vid, Video) or x.vid is None
                and isinstance(x.txt, Text) or x.txt is None
                and isinstance(x.aud, Audio) or x.aud is None
        ), "EEGDatasetDataPoint are required to not be in tensor form here."

        nid = x.entry_id + "_" + str(idx)
        y = EEGDatasetDataPoint(
            entry_id=nid,
            # No files path are associated now.
            eeg=replace(x.eeg, interval=segment, entry_id=nid),
            vid=replace(x.vid, interval=segment, entry_id=nid),
            aud=replace(x.aud, interval=segment, entry_id=nid),
            txt=replace(x.txt, entry_id=nid) if x.txt is not None else None,
        )

        if self.split_pipeline is not None:
            y: EEGDatasetDataPoint = call_pipelines(y, self.split_pipeline)

        # Save Video
        vid_out_path = out_folder + f'{y.entry_id}.mp4'
        y.vid.data.write_videofile(vid_out_path, audio=False, codec="libx264", ffmpeg_params=["-pix_fmt", "yuv420p"], )
        y.vid.file_path = os.path.relpath(Path(vid_out_path).resolve(), self.output_path)

        # Save Audio
        aud_out_path = out_folder + f'{y.entry_id}.wav'
        y.aud.data.write_audiofile(aud_out_path)
        y.aud.file_path = os.path.relpath(Path(aud_out_path).resolve(), self.output_path)

        # Save Text if exists
        if y.txt is not None and y.txt.data is not None:
            txt_out_path = out_folder + f'{y.entry_id}.txt'
            with open(txt_out_path, "w", encoding="utf-8") as f:
                f.write(y.txt.data)
            y.txt.file_path = os.path.relpath(Path(txt_out_path).resolve(), self.output_path)

        # EEG data is treated differently and aggregated to save space.
        # So it is not performed here.
        return y

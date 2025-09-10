import os.path
import traceback
from abc import abstractmethod, ABC
from dataclasses import replace
from pathlib import Path
from typing import Optional, Text

import numpy as np
import pandas as pd

from common.data.audio import Audio
from common.data.data_point import DatasetDataPoint, EEGDatasetDataPoint, EEGDatasetTransformWrapper, call_pipelines, \
    AgnosticDatasetPoint, AgnosticDatasetTransformWrapper
from common.data.eeg import EEG
from common.data.eeg.transforms import EEGToMneRawFromChannels
from common.data.loader import DataPointsLoader
from common.data.sampler import Segmenter
from common.data.video import Video

SPEC_FILE_NAME: str = "spec.csv"


class Preprocessor(ABC):
    def __init__(self, output_path: str):
        """
        Creates a processed dataset in a target folder. Info of the new ds are contained in the spec.csv
        """
        self.output_path: str = output_path

    @abstractmethod
    def preprocess(self, x: DatasetDataPoint) -> DatasetDataPoint | list[DatasetDataPoint]:
        pass

    def run(self, loader: DataPointsLoader) -> bool:
        try:
            # Read an existing spec if it was computed.
            existing_df: Optional[pd.DataFrame] = None
            existing_path = self.output_path + SPEC_FILE_NAME
            if Path(existing_path).exists():
                existing_df = pd.read_csv(existing_path)

            docs: list[DatasetDataPoint] = []

            # todo: If multithreading do it here on samples. Consigliano Queue per generare objects
            for i in loader.scan():
                key = i.get_identifier()
                if existing_df is not None and existing_df[key].str.contains(i.eid).any():
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


"""
    TODO: Era troppo bello pensare di fare una cosa cosi.
        Ci serve almeno una fn per salvare i record generati. -> Abstract method o comunque gestione di torch like structures che salvo in "database".
        Problema dello split in segmenti: Posso fare alcuni step prima?
        Problema: Estrazione testo da audio come faccio?
"""


class SegmenterPreprocessor(Preprocessor):
    def __init__(self, output_path: str, segmenter: Segmenter,
                 # In order to work with EEG data
                 ch_names: list[str], ch_types: list[str], pipeline: AgnosticDatasetTransformWrapper):
        super().__init__(output_path)
        self.segmenter: Segmenter = segmenter
        # todo AgnosticDatasetTransformWrapper
        self.pipeline: AgnosticDatasetTransformWrapper = pipeline
        # EEG mapping for mne
        self.ch_names: list[str] = ch_names
        self.ch_types: list[str] = ch_types

    def preprocess(self, x: AgnosticDatasetPoint) -> AgnosticDatasetPoint | list[AgnosticDatasetPoint]:
        original_sample_id = x.eid
        if not hasattr(x, EEG.modality_code()):
            raise ValueError("EEG data is required by design in any dataset")

        if x[EEG.modality_code()].data.shape[0] != len(self.ch_names):
            x[EEG.modality_code()].data = x[EEG.modality_code()].data.T  # Transpose
        assert x[EEG.modality_code()].data.shape[0] == len(self.ch_names), "Shape mismatch for EEG data"
        segments: list[tuple[int, int]] = self.segmenter.compute_segments(x[EEG.modality_code()])

        x_out_folder = self.output_path + x.eid + "/"
        Path(x_out_folder).mkdir(parents=True, exist_ok=True)
        x_segments = [self.preprocess_segment(x, idx, segment, x_out_folder) for idx, segment in enumerate(segments)]

        eeg_out_path: str = self.output_path + f'{original_sample_id}'
        x.export(eeg_out_path, self.output_path, only=EEG.modality_code())

        # EEG data are collected in files.
        for x_segment in x_segments:
            x_segment[EEG.modality_code()].file_path = x[EEG.modality_code()].file_path
        """
        TODO: Qualcosa del genere: MOLTO BUONO cosi
        torch.save({
            'eeg': torch.stack([sample['eeg'] for sample in a]),  # (17, 14, 8, 200)
            'vid': torch.stack([sample['vid'] for sample in a]),   # (17, 400)
            EFFICENTE anche cosi quindi posso tnere le mask e tutto
            'aud': {
                "in":torch.stack([sample['aud']["input_features"] for sample in a]),
                "attn": torch.stack([sample['aud']["attention_mask"] for sample in a])
            }
        }, 'batched_file.pt')
        """
        return x_segments

    def preprocess_segment(self, x: AgnosticDatasetPoint, idx: int,
                           segment: tuple[int | float | np.ndarray, int | float | np.ndarray], out_folder: str) \
            -> AgnosticDatasetPoint:
        if isinstance(segment[0], np.ndarray):
            segment = (segment[0].item(), segment[1].item())

        y = x.clone(x.eid + "_" + str(idx))
        for arg, value in y.__dict__.items():
            if hasattr(value, "interval"):
                value.__setattr__("interval", segment)

        if self.pipeline is not None:
            y = self.pipeline.call(y)

        y.export(out_folder, self.output_path, EEG.modality_code())
        return y


class EEGSegmenterPreprocessor(Preprocessor):
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

    def preprocess(self, x: EEGDatasetDataPoint) -> list[DatasetDataPoint]:
        # Process output from x (in samples) will be stored under /{output_path]/{x._entry_id}+SUFFIX
        original_sample_id = x.entry_id

        if x.eeg.data.shape[0] != len(self.ch_names):
            x.eeg.data = x.eeg.data.T  # Transpose

        assert x.eeg.data.shape[0] == len(self.ch_names), "Shape mismatch for EEG data"
        x.eeg = EEGToMneRawFromChannels(channel_names=self.ch_names, channel_types=self.ch_types)(x.eeg)
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

        return x_segments

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

import traceback
from abc import abstractmethod, ABC
from pathlib import Path
from typing import Optional, TypeVar, Generic

import numpy as np
import pandas as pd
import torch

from common.data.data_point import AgnosticDatasetPoint, AgnosticDatasetTransformWrapper
from common.data.eeg import EEG
from common.data.loader import DataPointsLoader
from common.data.sampler import Segmenter
from common.data.utils import build_tensor_dict, sanitize_for_ast

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
            for idx, (seg, segment) in enumerate(zip(x_segments, segments))
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
        objects = [s.to_dict() if hasattr(s, "to_dict") else s for s in segments]
        torch.save(build_tensor_dict(objects), output_path + ".pt")

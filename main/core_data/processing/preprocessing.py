import traceback
from abc import abstractmethod, ABC
from concurrent.futures import ThreadPoolExecutor
from itertools import batched
from pathlib import Path
from typing import Optional, TypeVar, Generic

import numpy as np
import pandas as pd
from tensordict import TensorDict, stack

from main.core_data.data_point import FlexibleDatasetPoint, FlexibleDatasetTransformWrapper
from main.core_data.loader import DataPointsLoader
from main.core_data.utils import sanitize_for_ast, timed
from main.utils.logging import make_logger

SPEC_FILE_NAME: str = "spec.csv"

T = TypeVar("T")


class Preprocessor(ABC, Generic[T]):
    def __init__(self, output_path: str):
        """
        Creates a processed dataset in a target folder. Info of the new dataset are contained in the spec.csv
        """
        self.output_path: str = output_path
        self.logger = make_logger(self.__class__.__name__)

    @abstractmethod
    def preprocess(self, x: T) -> dict | list[dict]:
        pass

    @abstractmethod
    def export(self, x: list[T], output_path: str) -> None:
        pass

    @timed()
    def run(self, loader: DataPointsLoader, workers: int = 1) -> bool:
        try:
            # Read an existing spec if it was computed.
            existing_df: Optional[pd.DataFrame] = None
            existing_path = self.output_path + SPEC_FILE_NAME
            Path(self.output_path).mkdir(parents=True, exist_ok=True)
            if Path(existing_path).exists():
                existing_df = pd.read_csv(existing_path)

            if workers > 1:
                for block in batched(loader.scan(), workers):
                    docs = []  # Where the stuff is stored.
                    with ThreadPoolExecutor(max_workers=workers) as executor:
                        for i in block:
                            key = i.get_identifier()
                            if existing_df is not None and (existing_df[key] == i.eid).any():
                                continue  # This element was already processed.
                            docs.append(executor.submit(self.preprocess, i))

                    for doc in docs:
                        df = pd.DataFrame([d for d in doc.result()])
                        if existing_df is not None:
                            df = pd.concat([df, existing_df], ignore_index=True)
                        df.to_csv(self.output_path + "spec.csv", index=False)
                        existing_df = df

                self.logger.info("Procedure finished correctly.")
                self.logger.info("Spec file can be found at:", self.output_path, "spec.csv")
                return True

            for i in loader.scan():
                key = i.get_identifier()
                if existing_df is not None and (existing_df[key] == i.eid).any():
                    continue  # This element was already processed.

                docs = [e for e in self.preprocess(i)]
                df = pd.DataFrame([d for d in docs])
                if existing_df is not None:
                    df = pd.concat([df, existing_df], ignore_index=True)

                df.to_csv(self.output_path + "spec.csv", index=False)
                existing_df = df

            self.logger.info("Procedure finished correctly.")
            self.logger.info("Spec file can be found at:", self.output_path, "spec.csv")
            return True

        except Exception as e:
            self.logger.critical("Preprocessing pipeline failed for an unexpected error:")
            self.logger.critical(e, exc_info=True)
            return False


class TorchExportsSegmentsReadyPreprocessor(Preprocessor[FlexibleDatasetPoint]):
    def __init__(self, output_path: str,
                 # Specs folder to give
                 extraction_data_folder: str,
                 # In order to work with EEG data
                 segment_pipeline: FlexibleDatasetTransformWrapper,
                 sample_pipeline: Optional[FlexibleDatasetTransformWrapper] = None):
        super().__init__(output_path)
        self.shared_pipeline: FlexibleDatasetTransformWrapper = sample_pipeline
        self.pipeline: FlexibleDatasetTransformWrapper = segment_pipeline
        self.extraction_data_folder: str = extraction_data_folder

    @timed()
    def preprocess(self, x: FlexibleDatasetPoint) -> dict | list[dict]:
        segments = pd.read_csv(self.extraction_data_folder + str(x.eid) + "-segments.csv").to_dict(orient="records")
        if self.shared_pipeline is not None:
            x = self.shared_pipeline.call(x, keep_type=True)

        x_segments = []
        total_elements = len(segments)
        for idx, segment in enumerate(segments):
            self.logger.info(f"About to process the element {idx + 1}/{total_elements} for {x.eid}")
            self.preprocess_segment(x, (segment["start"], segment["stop"]))

        output_path: str = self.output_path + f'{x.eid}'
        self.export(x_segments, output_path)
        # Return file specification
        return_segments = [
            {"index": idx, x.get_identifier(): x.eid, "segment": segment}
            for idx, (seg, segment) in enumerate(zip(x_segments, segments))
        ]
        return_segments = sanitize_for_ast(return_segments)
        return return_segments

    @timed(suppress_timed=False)
    def preprocess_segment(self, x: FlexibleDatasetPoint,
                           segment: tuple[int | float | np.ndarray, int | float | np.ndarray],
                           ) -> FlexibleDatasetPoint:
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

    def export(self, x: list[FlexibleDatasetPoint], output_path: str) -> None:
        objects = [TensorDict(s.to_dict()) if hasattr(s, "to_dict") else TensorDict(s) for s in x]
        tensor_dict = stack(objects, dim=0)
        Path(output_path).mkdir(parents=True, exist_ok=True)
        tensor_dict.memmap(output_path)

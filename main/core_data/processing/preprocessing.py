import math
from abc import abstractmethod, ABC
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from itertools import batched
from pathlib import Path
from typing import Optional, TypeVar, Generic, Literal

import numpy as np
import pandas as pd
from tensordict import TensorDict, stack
from tqdm import tqdm

from main.core_data.data_point import FlexibleDatasetPoint, FlexibleDatasetTransformWrapper
from main.core_data.loader import DataPointsLoader
from main.core_data.media.metadata.metadata import Metadata
from main.core_data.quantization import Float16ToInt8Quantizer
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
    def run(self, loader: DataPointsLoader) -> bool:
        try:
            # Read an existing spec if it was computed.
            existing_df: Optional[pd.DataFrame] = None
            existing_path = self.output_path + SPEC_FILE_NAME
            Path(self.output_path).mkdir(parents=True, exist_ok=True)
            if Path(existing_path).exists():
                existing_df = pd.read_csv(existing_path)
            total: int = len(loader)
            counter: int = 0
            for i in tqdm(loader.scan(), total=total, desc="Processing"):
                key = i.get_identifier()
                if existing_df is not None and (existing_df[key] == i.eid).any():
                    tqdm.write(f"Element {key} will be skipped as it was already processed.")
                    continue  # This element was already processed.

                docs = [e for e in self.preprocess(i)]
                df = pd.DataFrame([d for d in docs])
                if existing_df is not None:
                    df = pd.concat([df, existing_df], ignore_index=True)

                df.to_csv(self.output_path + "spec.csv", index=False)
                existing_df = df
                counter += 1
                self.logger.info(f"Processed {counter}/{total}")

            self.logger.info("Procedure finished correctly.")
            self.logger.info(f"Spec file can be found at:{self.output_path} spec.csv")
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
            x_segments.append(self.preprocess_segment(x, (segment["start"], segment["stop"])))

        output_path: str = self.output_path + f'{x.eid}'
        self.export(x_segments, output_path)

        # Return file specification
        base_object = {}
        if "meta" in x:
            # We have metaobject to pass to the csv. Better have it redundant than not enough.
            base_object = {key: value for key, value in asdict(x.meta.data).items()}

        return_segments = [
            base_object | {"index": idx, x.get_identifier(): x.eid, "segment": segment}
            for idx, (seg, segment) in enumerate(zip(x_segments, segments))
        ]
        return_segments = sanitize_for_ast(return_segments)
        return return_segments

    @timed(suppress=False)
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
        # todo apply quantizartion + export to h5 with teacher sample
        objects = [
            TensorDict(s.to_dict()) if hasattr(s, "to_dict") else TensorDict(s) for s in x
        ]

        tensor_dict = stack(objects, dim=0)
        Path(output_path).mkdir(parents=True, exist_ok=True)
        tensor_dict.memmap(output_path)


class TorchExportsKdSegmentsReadyPreprocessor(Preprocessor[FlexibleDatasetPoint]):
    def __init__(self, output_path: str,
                 # Specs folder to give
                 extraction_data_folder: str,
                 # In order to work with EEG data
                 student_segment_pipeline: FlexibleDatasetTransformWrapper,
                 teacher_segment_pipeline: FlexibleDatasetTransformWrapper,
                 student_sample_pipeline: Optional[FlexibleDatasetTransformWrapper] = None,
                 teacher_sample_pipeline: Optional[FlexibleDatasetTransformWrapper] = None,
                 quantization_keys: list[str] = (),
                 ):
        super().__init__(output_path)
        self.student_shared_pipeline: FlexibleDatasetTransformWrapper = student_sample_pipeline
        self.student_pipeline: FlexibleDatasetTransformWrapper = student_segment_pipeline

        self.teacher_shared_pipeline: FlexibleDatasetTransformWrapper = teacher_sample_pipeline
        self.teacher_pipeline: FlexibleDatasetTransformWrapper = teacher_segment_pipeline
        self.extraction_data_folder: str = extraction_data_folder
        self.quantizer = Float16ToInt8Quantizer()
        self.quantization_keys: list[str] = quantization_keys

    @timed()
    def preprocess(self, x: FlexibleDatasetPoint) -> dict | list[dict]:
        segments = pd.read_csv(self.extraction_data_folder + str(x.eid) + "-segments.csv").to_dict(orient="records")
        student_x, teacher_x = x, x
        if self.student_shared_pipeline is not None:
            student_x = self.student_shared_pipeline.call(x, keep_type=True)
        if self.teacher_shared_pipeline is not None:
            teacher_x = self.teacher_shared_pipeline.call(x, keep_type=True)

        student_x_segments = []
        teacher_x_segments = []
        total_elements = len(segments)
        for idx, segment in enumerate(segments):
            self.logger.info(f"About to process the element {idx + 1}/{total_elements} for {student_x.eid}")
            student_segment = self.preprocess_segment(student_x, (segment["start"], segment["stop"]), "student")
            student_x_segments.append(student_segment)

            teacher_segment = self.preprocess_segment(teacher_x, (segment["start"], segment["stop"]), "teacher")
            teacher_x_segments.append(teacher_segment)

        out_path = Path(self.output_path) / str(x.eid)
        self.export(student=student_x_segments, teacher=teacher_x_segments, output_path=out_path)

        # Return file specification
        base_object = {}
        if hasattr(student_x, Metadata.modality_code()):
            # We have metaobject to pass to the csv. Better have it redundant than not enough.
            base_object = {key: value for key, value in student_x.meta.data.items()}

        return_segments = [
            base_object | {"index": idx, student_x.get_identifier(): student_x.eid, "interval": segment}
            for idx, (seg, segment) in enumerate(zip(student_x_segments, segments))
        ]

        return_segments = sanitize_for_ast(return_segments)

        return return_segments

    @timed(suppress=False)
    def preprocess_segment(self, x: FlexibleDatasetPoint,
                           segment: tuple[int | float | np.ndarray, int | float | np.ndarray],
                           target: Literal['student', 'teacher']
                           ) -> FlexibleDatasetPoint:
        if isinstance(segment[0], np.ndarray):
            segment = (segment[0].item(), segment[1].item())

        y = x.clone(x.eid)  # entry_id is useless for this approach
        for arg, value in y.__dict__.items():
            if hasattr(value, "interval"):
                value.__setattr__("interval", segment)

        if self.student_pipeline is None:
            raise ValueError("pipeline is required for preprocessing")

        pipeline = self.teacher_pipeline if target == "teacher" else self.student_pipeline
        y = pipeline.call(y)

        return y

    def export(self, output_path: Path, **x: list[FlexibleDatasetPoint]) -> None:
        return_object = {}
        # todo promote meta as they are identical

        optional_meta = None

        for key, value in x.items():
            objects = [TensorDict(s.to_dict()) if hasattr(s, "to_dict") else TensorDict(s) for s in value]
            td = stack(objects, dim=0)

            if Metadata.modality_code() in td:
                if optional_meta is not None and not (optional_meta == td["meta"]).all():
                    raise ValueError("meta does not match with previous meta found")
                optional_meta = td.pop(Metadata.modality_code())
            return_object[key] = td

        if optional_meta is not None:
            return_object[Metadata.modality_code()] = optional_meta

        # now we add h5 and we are done here
        output_path.mkdir(parents=True, exist_ok=True)
        bs = len(next(iter(x.values())))
        TensorDict(return_object, batch_size=bs).memmap(str(output_path))

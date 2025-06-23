from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd


class MediaPreProcessingPipeline(ABC):
    @abstractmethod
    def process(self, media: list | np.ndarray | str):
        pass

    @abstractmethod
    def process_output_shape(self) -> tuple:
        pass


RAW_KEY = "raw"
PROCESSED_KEY = "processed"
METADATA_KEY = "metadata"


# Handles the data collection relative to a media.
class MediaCollector(ABC):
    def __init__(self, processor: MediaPreProcessingPipeline):
        self.media_collection = {
            # Raw has to be initialized.
            RAW_KEY: None, PROCESSED_KEY: np.empty((0,) + processor.process_output_shape()), METADATA_KEY: []
        }
        self.processor = processor

    def get_media(self, index: int) -> dict:
        raw, processed, meta = self._data_as_tuple()

        if not (0 <= index < len(raw)):
            raise IndexError("Searched index is not a valid element!")

        if processed[index] is None:
            processed[index] = self.processor.process(raw[index].squeeze())

        return {
            RAW_KEY: raw[index], PROCESSED_KEY: processed[index],
            METADATA_KEY: meta[index] if 0 <= index < len(meta) else None,
        }

    def _update_data(self, raw: np.ndarray, processed: np.ndarray, meta: list):
        m = self.media_collection
        m[RAW_KEY], m[PROCESSED_KEY], m[METADATA_KEY] = raw, processed, meta

    def __len__(self):
        return self.media_collection[RAW_KEY].shape[0]

    def _data_as_tuple(self) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None]:
        m = self.media_collection
        return m[RAW_KEY], m[PROCESSED_KEY], m[METADATA_KEY]

    def get_processed_data(self):
        return self.media_collection[PROCESSED_KEY]

    def get_info(self, index: int) -> dict:
        raw, processed, meta = self._data_as_tuple()
        if 0 <= index < len(raw):
            raise IndexError("Searched index is not a valid element!")

        return {"processed": 0 <= index < len(processed)} | (
            meta[index].to_dict() if 0 <= index < len(meta) else {}
        )

    def __call__(self, index: int, *args, **kwargs):
        return self.get_media(index)

    @abstractmethod
    def load_additional_resource(self, file_path: str, **kwargs):
        pass


class FileReferenceMediaCollector(MediaCollector, ABC):
    # File reference handles 1 item at the time, pretty simple if you ask me!
    def load_additional_resource(self, file_path: str, **kwargs):
        # Add the new row for the resource.
        # Processing is not handled on load so lazy init it is
        raw, processed, meta = self._data_as_tuple()

        # Update the raw
        if raw is None:
            raw = np.empty((0,))
        raw = np.concatenate((raw, np.array([file_path])))

        # Create empty rows for processed
        empty = np.empty((1,) + self.processor.process_output_shape())
        processed = np.concatenate((processed, empty))

        # Add a metadata dictionary
        meta.append(dict())
        self._update_data(raw, processed, meta)


class NumpyDataMediaCollector(MediaCollector, ABC):
    def __init__(self, bonus_info_args: list[str], processor: MediaPreProcessingPipeline):
        super().__init__(processor)
        self.bonus_info_args = bonus_info_args

    def load_additional_resource(self, file_path: str, **kwargs):
        raw, processed, meta = self._data_as_tuple()

        resource = np.load(file_path, allow_pickle=True)
        raw_data = np.array([resource[k] for k in resource.files]).T
        repeats = raw_data.shape[0]

        if raw is None:
            correct_shape = list(raw_data.shape)
            correct_shape[0] = 0  # Empty initialization for concat requires 0 rows.
            raw = np.empty(tuple(correct_shape))

        raw = np.concatenate((raw, raw_data))

        # Add empty rows for the processed-data
        empty = np.empty((repeats,) + self.processor.process_output_shape())
        processed = np.concatenate((processed, empty))

        # Add custom metadata
        metadata_row = {col: kwargs[col] for col in self.bonus_info_args}
        metadata_row["raw_dimensions"] = resource.files
        [meta.append(metadata_row) for i in range(repeats)]
        self._update_data(raw, processed, meta)


class PandasCsvDataMediaCollector(MediaCollector, ABC):
    def __init__(self, bonus_info_args: list[str], processor: MediaPreProcessingPipeline):
        super().__init__(processor)
        self.bonus_info_args = bonus_info_args

    def load_additional_resource(self, file_path: str, **kwargs):
        raw, processed, meta = self._data_as_tuple()

        df = pd.read_csv(file_path)
        raw = df.to_numpy() if raw.size == 0 else np.concatenate((raw, df.to_numpy()))
        repeats = len(df)

        # Add empty rows for the processed-data
        empty = np.empty((repeats,) + self.processor.process_output_shape())
        processed = np.concatenate((processed, empty))

        # Add custom metadata
        metadata_row = {col: kwargs[col] for col in self.bonus_info_args}
        metadata_row["raw_dimensions"] = list(df.columns)
        [meta.append(metadata_row) for i in range(len(df))]
        self._update_data(raw, processed, meta)

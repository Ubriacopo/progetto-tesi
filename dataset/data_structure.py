# TODO Lavora su questi nel fork di VATE
# Da VATE mi serve per capire come funziona.
# Immagino serva una sua implementazione per EEG
import os
from abc import ABC, abstractmethod
from pathlib import Path

import pandas as pd

from dependencies.VATE.dataset import DatasetFS
from dependencies.VATE.utils import verbatimT, verbatimO


class Media(ABC):
    def __init__(self, config: dict, dataset: DatasetFS, filename: str, store: bool,
                 store_info: bool = True, verbose: int = 0):
        self.configuration = config
        self.dataset = dataset
        self.filename = filename
        self.verbose = verbose

        self.store_info = store_info
        if store_info:
            self.merge_info_dataset()

        self.store_media_data() if store else self.load_media_data()

    def info(self):
        """
        Dataset description.
        """
        verbatimT(1, 1, f"Dataset: {self.configuration["DATASET_NAME"]}")
        verbatimT(1, 1, f"size: {self.dataset.size()}:", deep=1)
        verbatimT(1, 1, f"series: {list(self.dataset.data_frame.keys())}:", deep=1)
        verbatimT(1, 1, "done.\n")

    def set_dataset(self, dataset: DatasetFS, store_info=True) -> None:
        """
        Sets a new dataset to process for the Media object.
        """
        self.dataset = dataset
        self.store_info = store_info
        if store_info:
            self.merge_info_dataset()

    def store_media_data(self):
        output_path = os.path.join(self.configuration['ROOT'], self.configuration["OUTPUT_DIR"], self.filename)
        self.dataset.data_frame.to_pickle(output_path)
        verbatimT(self.verbose, 1, "Media descriptor stored into file: " + str(output_path))

    def load_media_data(self):
        assert self.filename is not None, "Filename must be provided"
        assert self.filename.endswith(".pkl"), "Filename must be a .pkl file"
        output_path = os.path.join(self.configuration['ROOT'], self.configuration["OUTPUT_DIR"], self.filename)
        if not Path(str(output_path)).exists():
            return

        self.dataset.data_frame = pd.read_pickle(str(output_path))
        verbatimT(self.verbose, 1, "Dataset descriptor loaded from file: " + str(output_path))
        verbatimO(self.verbose, 1, self.dataset.data_frame)

    @abstractmethod
    def merge_info_dataset(self):
        """
        Merges all Media info in the dataset Dataset as a dict.
        The main container to merge is self.data_frame
        """
        pass

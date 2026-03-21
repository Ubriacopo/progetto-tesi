from pathlib import Path

import pandas as pd
import tensordict
from torch.utils.data import Dataset

from main.core_data.media.assessment.assessment import Assessment
from main.utils.logging import make_logger


class TdSegmentedExperimentDataset(Dataset):
    def __init__(self, dataset_path: str, dataset_spec_file: str, accessible_user_ids: list[int]):
        """

        :param dataset_path: Where the dataset is stored
        :param dataset_spec_file:  The dataset spec file
        :param accessible_user_ids: Accessible user ids. The ones I can read (this avoids leaking subjects from train/test)
        """
        self.logger = make_logger(self.__class__.__name__)
        self.dataset_path: Path = Path(dataset_path)
        self.df = pd.read_csv(dataset_spec_file, index_col=False)

        self.df = self.df[self.df["person_id"].isin(accessible_user_ids)]
        self.df = self.df.drop_duplicates(subset="experiment")

    def __getitem__(self, idx: int):
        sample = self.df.iloc[idx].to_dict()
        td = tensordict.load_memmap(self.dataset_path / str(sample["experiment"]))
        return td

    def __len__(self):
        return self.df.shape[0]

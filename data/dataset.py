from abc import abstractmethod

import torch.utils.data


# todo up next
# todo to augment I could work on the different 3 channels of input separately
# Dataset
class EEGDataset(torch.utils.data.Dataset):
    def initialize(self, files_root_path: str):
        pass

    def info(self):
        """
        Prints the dataset description in verbatim
        """
        pass

    @abstractmethod
    def scan(self):
        """
        Scan the filesystem pointed by datapath and retrieves the valid resources.
        Might not be necessary.
        :return:
        """
        pass

    @abstractmethod
    def __getitem__(self, item):
        pass

    @abstractmethod
    def len(self):
        pass


class AMIGOSDataset(EEGDataset):
    # Own naming convention etc here
    def __init__(self, eeg_data_path: str, video_path: str):
        pass


class DREAMERDataset(EEGDataset):
    pass


class DEAPDataset(EEGDataset):
    pass

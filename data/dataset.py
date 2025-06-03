from abc import abstractmethod

import torch.utils.data


# todo up next
# todo to augment I could work on the different 3 channels of input separately
# Dataset
class EEGDataset(torch.utils.data.Dataset):
    def initialize(self, files_root_path: str):
        pass

    def _extract_video(self):
        pass

    def _extract_audio(self):
        pass

    def _extract_text(self):
        pass

    def _extract_eeg(self):
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
    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        # Returns a video, an audio, a text transcript and an eeg signal
        # TODO: Need a trascript tool? Or better yet. We know exactly the timing of movie so we can know
        # the correct script by downloading it.
        pass

    @abstractmethod
    def len(self):
        pass


# VATE is trained on frontal data. Face_video are most prolly the best to work on with this knowledge.
# I could also try to exploit the Depth videos?
# Each dataset has its own rigid data structure
class AMIGOSDataset(EEGDataset):
    def scan(self):
        pass

    def source_is_valid(self):
        return True

    def __init__(self, base_path: str):
        self.base_path = base_path
        if not self.source_is_valid():
            raise FileNotFoundError("The given dataset doesn't exist or is an invalid instance of AMIGOS.")


class DREAMERDataset(EEGDataset):
    pass


class DEAPDataset(EEGDataset):
    pass

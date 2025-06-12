from abc import abstractmethod

import pandas as pd
import torch.utils.data

from data.media.signal import SignalDataCollector, Signal


# TODO: Modality masks. I must allow to pass only some types of data.

class DataListing:
    pass


class AMIGOSDataListing(DataListing):
    def __init__(self, resource_file: str):
        self.resource_file = resource_file if not resource_file == "" else "../resources/Metadata_xlsx/Experiment_Data.xlsx"
        self.listing: pd.DataFrame = pd.read_excel(self.resource_file)


class DREAMERDataListing(DataListing):
    def __init__(self, resource_file: str):
        self.resource_file = resource_file if not resource_file == "" else "../resources/Metadata_xlsx/Experiment_Data.xlsx"


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
    def pickle(self, output_path: str):
        pass

    @abstractmethod
    def source_is_valid(self):
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
    def __init__(self, base_path: str, signal_collector: SignalDataCollector):
        self.base_path = base_path

        # Where signal data is designed to be stored
        self.signal_collector = signal_collector
        self.listing: pd.DataFrame | None = None
        if not self.source_is_valid():
            raise FileNotFoundError("The given dataset doesn't exist or is an invalid instance of AMIGOS.")

    def pickle(self, output_path: str):
        pass

    def scan(self):
        # Scans the experiment data to have all required information to get a sample.
        resource_file = self.base_path + "../Metadata_xlsx/Experiment_Data.xlsx"
        sheets = pd.read_excel(resource_file, sheet_name=None)

        df = sheets["Short_Video_Order"]
        # Replace trailing spaces
        df.columns = df.columns.str.replace(r'[()]', '', regex=True).str.replace(' ', '_')

        num_columns = [col for col in df.columns if "Number" in col]
        id_columns = [col for col in df.columns if "VideoID" in col]

        df_num = df.melt(id_vars=['Exp1_ID', 'UserID'], value_vars=num_columns,
                         var_name='Trial', value_name='Video_Number')
        df_num["Trial_Num"] = df_num['Trial'].str.extract(r'(\d+)')
        df_num = df_num.drop('Trial', axis=1)

        df_id = df.melt(id_vars=['Exp1_ID', 'UserID'], value_vars=id_columns,
                        var_name='Trial', value_name='Video_ID')
        df_id["Trial_Num"] = df_id['Trial'].str.extract(r'(\d+)')
        df_id = df_id.drop('Trial', axis=1)

        # We want each row to be a specific experiment
        self.listing = pd.merge(df_num, df_id, on=['Exp1_ID', 'UserID', 'Trial_Num'])

    def len(self):
        pass

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        # TODO: I need an array of PID + Video indexed
        row = self.listing.iloc[index]
        # I have to return: a Video, an Audio, a Text and a Signal data

        # The data of the selected row:
        uid = row["UserID"]
        trial_num = row["Trial_Num"]

        signal = Signal(self.signal_collector, uid, self.base_path, trial_num)

        return signal()

    def source_is_valid(self):
        # Check folder structure to be correct
        return True


class DEAPDataset(EEGDataset):
    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        pass

    def scan(self):
        pass

    def len(self):
        pass


class DREAMERDataset(EEGDataset):
    def scan(self):
        pass

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        pass

    def len(self):
        pass

import numpy as np
from scipy.io import loadmat
from torch import Tensor

from data.media.media import Media


# The signal works on a file but contains more results.
class Signal(Media):
    """
        I won't for sure be able to read the full dataframe immediately.
        What I can do is load the entire file. Should I?
    """

    def __init__(self, file_path: str, lazy: bool = True):
        self.video_id: str | None = None
        self.data: Tensor | None = None
        self.labels_ext_annotation: Tensor | None = None
        self.labels_self_assessment: Tensor | None = None
        super().__init__(file_path, lazy)

    def get_info(self):
        return {"file_path": self.file_path, "frequency": ""}

    def _inner_load(self, **kwargs):
        numpy_object = np.load(self.file_path, allow_pickle=True)
        self.video_id = numpy_object["VideoIDs"]
        # This info will for sure be needed.
        self.data = numpy_object["joined_data"]

        # These labels can be used to evaluate the model training ?
        self.labels_self_assessment = numpy_object["labels_selfassessment"]
        self.labels_ext_annotation = numpy_object["labels_ext_annotation"]

    def _inner_process(self, **kwargs):
        pass


def extract_trial_data(destination_path: str, participant: str, source_path: str):
    """
    Process the files so that each trial is split. We keep naming conventions to track the partecipant.
    The data is converted to a numpy friendly type to help us work better (we have some advantages).
    This is mostly to extract Signal instances without having to read too many files.
    Downside could be having many I/O operations. We see how it goes.

    TODO: We could also avoid splitting in many files and do less than the ones we have.
            (Hoping RAM to be much). And rotate the files in case.
            Videos are obviously loaded a batch at the time. (Ill check VATE implementation to see how he who knows did it)

    :param destination_path:
    :param participant:
    :param source_path:
    :return:
    """
    #   Idea is to process like this (For faster loading):
    #       - Every trial is split to create a single file.
    #       - This way we just have to store in name percipient-trial_num.npy
    #
    mat = loadmat(source_path)  # Source file

    data = {k: v for k, v in mat.items() if not k.startswith("__")}

    for key in data:
        # Remove the heading dimension
        data[key] = data[key].squeeze()

    num_trials = data["joined_data"].shape[0]
    for i in range(num_trials):
        # For each trial we create a file. Now we should name it in an intelligent way.
        trial = {key: data[key][i] for key in data}

        # TODO: We should work on this. to incorporate: participant ID (from name P01)
        #           + Video ID could also be integrated oppure incrementale diretto di trial.
        filename = f"{destination_path}/trial_{i}_P{participant}.npz"
        np.savez_compressed(filename, **trial)

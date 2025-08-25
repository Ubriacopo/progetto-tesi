import pickle
from abc import abstractmethod, ABC
from pathlib import Path

import numpy as np
import torch.utils.data

from common.data.media import MediaCollector, PROCESSED_KEY, FileReferenceMediaCollector, PandasCsvDataMediaCollector, \
    NumpyDataMediaCollector
from common.data.preprocessing import MediaPreProcessingPipeline


class SimpleLoaderDataset(torch.utils.data.Dataset):
    def __init__(self, folds: list[list]):
        self.folds = folds

    def __getitem__(self, item: int):
        # Se manca un testo posso ricevere [num, num, None, num]. Multimodal safe?
        return tuple(lst[item] if len(lst) > item else None for lst in self.folds)

    def __len__(self):
        return len(self.folds[0])


# TODO: Modality masks. I must allow to pass only some types of data.
# todo to augment I could work on the different 3 channels of input separately
# Dataset
class EEGDataset(torch.utils.data.Dataset, ABC):
    base_path: str

    def __init__(self, signal_collector: MediaCollector, video_collector: MediaCollector,
                 audio_collector: MediaCollector, text_collector: MediaCollector, base_path: str):
        self.signal_collector: MediaCollector = signal_collector
        self.video_collector: MediaCollector = video_collector
        self.audio_collector: MediaCollector = audio_collector
        self.text_collector: MediaCollector = text_collector

        self.base_path = base_path
        self.scan()

    @abstractmethod
    def scan(self):
        """
        Scan the filesystem pointed by datapath and retrieves the valid resources.
        Might not be necessary.
        :return:
        """
        pass

    def restore(self, reference_path: str):
        with open(reference_path, 'rb') as handle:
            train_loader: SimpleLoaderDataset = pickle.load(handle)
            # TODO: Now we have the processed data. We should just set it to the media collector.
            #       What about the non processed data? How do i restore that one? Should I?
            #       I could be just storing the metadata with the processed data?
            self.video_collector.media_collection[PROCESSED_KEY] = train_loader.folds[0]
            self.audio_collector.media_collection[PROCESSED_KEY] = train_loader.folds[1]
            self.text_collector.media_collection[PROCESSED_KEY] = train_loader.folds[2]
            self.signal_collector.media_collection[PROCESSED_KEY] = train_loader.folds[3]

    def dump(self, output_path: str):
        vc, ac, tc, sc = self.video_collector, self.audio_collector, self.text_collector, self.signal_collector
        # Mah questo però richiede che il dataset sia sempre omogeneo.
        # Dovrei forse trovare un altro modo?
        simple = SimpleLoaderDataset(
            [vc.get_processed_data(), ac.get_processed_data(), tc.get_processed_data(), sc.get_processed_data()]
        )

        with open(output_path, 'wb') as handle:
            # noinspection PyTypeChecker
            pickle.dump(simple, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        v = self.video_collector.get_media(index)[PROCESSED_KEY]
        a = self.audio_collector.get_media(index)[PROCESSED_KEY]
        t = self.text_collector.get_media(index)[PROCESSED_KEY]
        s = self.signal_collector.get_media(index)[PROCESSED_KEY]
        return v, a, t, s

    def __len__(self):
        vc, ac, tc, sc = self.video_collector, self.audio_collector, self.text_collector, self.signal_collector
        return max(len(vc), len(ac), len(tc), len(sc))


# vate is trained on frontal data. Face_video are most prolly the best to work on with this knowledge.
# I could also try to exploit the Depth videos?
# Each dataset has its own rigid data structure
class AMIGOSDataset(EEGDataset):
    def __init__(self, signal_processor: MediaPreProcessingPipeline,
                 video_processor: MediaPreProcessingPipeline,
                 audio_processor: MediaPreProcessingPipeline,
                 text_processor: MediaPreProcessingPipeline, base_path: str):
        super().__init__(
            NumpyDataMediaCollector([], signal_processor),
            FileReferenceMediaCollector(video_processor),
            FileReferenceMediaCollector(audio_processor),
            PandasCsvDataMediaCollector([], text_processor),
            base_path
        )

    # Create the samples from one
    def handle_resource(self, file: Path, processed_data: np.ndarray):
        # P1_5_face.mov (Filename of face)
        # P(10,12,11,15)_B1_face.mov Of experiment congiunto?
        filename = file.name

    def load_participant_data(self, data_path: Path) -> dict:
        return_list = {}
        for f in data_path.iterdir():

            if not f.is_file() or f.suffix != '.npz':
                continue

            filename = f.stem
            return_list[filename] = {filename.split('_')[-1]: np.load(f, allow_pickle=True)}

        return return_list

    def scan(self):
        # Scans the experiment data to have all required information to load.
        # Processing happens on demand
        # Scan the processed Signal data:

        # Metadata:
        metadata_folder = self.base_path + "/metadata/"

        # Video files name: PXX_ (Person number)
        # Face videos:
        face_video_folder = self.base_path + "/face/"
        face_folder = Path(face_video_folder)

        for file in face_folder.iterdir():
            filename = file.name

        # Pre_Processed data structure:
        # Mi conviene dividere i file

        # Scan the videos
        # This yields video, audio and text -> No text actually (Paths not actual data)

        # Scan the processed data (.mat -> .npz) (Paths)

        return


class DEAPDataset(EEGDataset):
    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        pass

    def scan(self):
        pass


class DREAMERDataset(EEGDataset):
    def scan(self):
        pass

    def len(self):
        pass

# todo collector dataset

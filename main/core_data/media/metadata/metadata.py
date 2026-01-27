import dataclasses
from typing import Optional

from main.core_data.media import Media


@dataclasses.dataclass
class MetaObject:
    experiment: str | int # Unique experiment id (eid)
    dataset_id: str | int # The id of the dataset the sample belongs to
    person_id: str | int # id of the person of the experiment
    trial: str | int # Trial identifier that composes up the experiment by tuple (person_id, trial)


@dataclasses.dataclass
class Metadata(Media):
    @staticmethod
    def modality_code() -> str:
        return 'meta'

    def export(self, base_path: str, output_path_to_relative: str = None):
        pass

    interval: Optional[tuple[int, int]] = None

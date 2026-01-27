import dataclasses
from typing import Optional

from main.core_data.media import Media


@dataclasses.dataclass
class MetaObject:
    experiment: str | int
    dataset_id: str | int
    person_id: str | int
    trial: str | int


@dataclasses.dataclass
class Metadata(Media):
    @staticmethod
    def modality_code() -> str:
        return 'meta'

    def export(self, base_path: str, output_path_to_relative: str = None):
        pass

    interval: Optional[tuple[int, int]] = None

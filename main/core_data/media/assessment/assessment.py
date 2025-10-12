import dataclasses

from main.core_data.media import Media


@dataclasses.dataclass
class Assessment(Media):
    @staticmethod
    def modality_code() -> str:
        return "assessment"

    def export(self, base_path: str, output_path_to_relative: str = None):
        pass

    valence_idx: int = 0
    arousal_idx: int = 1
    dominance_idx: int = 2
    liking_idx: int = 3

import dataclasses

from main.core_data.media import Media


@dataclasses.dataclass
class Assessment(Media):
    @staticmethod
    def default_order():
        return 'v a d l'

    @staticmethod
    def modality_code() -> str:
        return "assessment"

    def export(self, base_path: str, output_path_to_relative: str = None):
        pass

import dataclasses

from main.core_data.media import Media


@dataclasses.dataclass
class Assessment(Media):
    rating_scales: list[tuple[int | float, int | float]]
    labels: list[str]

    @staticmethod
    def modality_code() -> str:
        return "assessment"

    def export(self, base_path: str, output_path_to_relative: str = None):
        print("AAAAAAAAAAAAAA")
        pass  # Cosi vedo se viene usato o posso rimuoverlo

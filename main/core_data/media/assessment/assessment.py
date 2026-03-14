import dataclasses

from main.core_data.media import Media


@dataclasses.dataclass
class Assessment(Media):
    rating_scale: tuple[int | float, int | float]

    @staticmethod
    def modality_code() -> str:
        return "assessment"

    def export(self, base_path: str, output_path_to_relative: str = None):
        print("AAAAAAAAAAAAAA")
        pass  # Cosi vedo se viene usato o posso rimuoverlo


@dataclasses.dataclass
class Valence(Assessment):
    @staticmethod
    def modality_code() -> str:
        return "valence"


@dataclasses.dataclass
class Arousal(Assessment):
    @staticmethod
    def modality_code() -> str:
        return "arousal"


@dataclasses.dataclass
class Dominance(Assessment):
    @staticmethod
    def modality_code() -> str:
        return "dominance"

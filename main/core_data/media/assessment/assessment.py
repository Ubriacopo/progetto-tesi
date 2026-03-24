import dataclasses
from typing import Final

from main.core_data.media import Media


class AssessmentLabels:
    AROUSAL: Final[str] = "arousal"
    VALENCE: Final[str] = "valence"
    DOMINANCE: Final[str] = "dominance"
    LIKING: Final[str] = "liking"
    FAMILIARITY: Final[str] = "familiarity"
    NEUTRAL: Final[str] = "neutral"
    DISGUST: Final[str] = "disgust"
    HAPPINESS: Final[str] = "happiness"
    SURPRISE: Final[str] = "surprise"
    ANGER: Final[str] = "anger"
    FEAR: Final[str] = "fear"
    SADNESS: Final[str] = "sadness"

    @classmethod
    def default_order(cls):
        return [
            cls.AROUSAL,
            cls.VALENCE,
            cls.DOMINANCE,
            cls.LIKING,
            cls.FAMILIARITY,
            cls.NEUTRAL,
            cls.DISGUST,
            cls.HAPPINESS,
            cls.SURPRISE,
            cls.ANGER,
            cls.FEAR,
            cls.SADNESS
        ]


@dataclasses.dataclass
class Assessment(Media):
    scales: list[tuple[int | float, int | float]]
    labels: list[str]

    @staticmethod
    def modality_code() -> str:
        return "assessment"

    def export(self, base_path: str, output_path_to_relative: str = None):
        print("AAAAAAAAAAAAAA")
        pass  # Cosi vedo se viene usato o posso rimuoverlo

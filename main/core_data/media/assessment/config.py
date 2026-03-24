import dataclasses


@dataclasses.dataclass
class ScoreLabelsConfig:
    labels: set[str]
    scales: set[tuple[int | float, int | float]]
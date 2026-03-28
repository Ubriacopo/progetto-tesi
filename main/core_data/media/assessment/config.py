import dataclasses


@dataclasses.dataclass
class ScoreLabelsConfig:
    labels: list[str]
    scales: list[tuple[int | float, int | float]]

@dataclasses.dataclass
class CategoricalLabelsConfig:
    labels: list[str]
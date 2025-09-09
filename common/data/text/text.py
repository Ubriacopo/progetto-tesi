import dataclasses

from common.data.media import Media


@dataclasses.dataclass
class Text(Media):
    @staticmethod
    def modality_code() -> str:
        return "txt"

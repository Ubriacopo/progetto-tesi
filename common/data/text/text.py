import dataclasses

from common.data.media import Media


@dataclasses.dataclass
class Text(Media):
    def modality_prefix(self) -> str:
        return "txt"

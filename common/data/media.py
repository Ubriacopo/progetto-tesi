import dataclasses
from abc import abstractmethod, ABC
from typing import Any, Optional


@dataclasses.dataclass
class Media(ABC):
    data: Any
    file_path: Optional[str]

    # Metadata is relative to media type.

    @abstractmethod
    def modality_prefix(self) -> str:
        pass

    def to_dict(self) -> dict:
        attrs = [f.name for f in dataclasses.fields(self)]
        metadata = {attr: getattr(self, attr) for attr in attrs if attr != "data" and attr != "file_path"}
        return {f"{self.modality_prefix()}_path": self.file_path, f"{self.modality_prefix()}_metadata": metadata}

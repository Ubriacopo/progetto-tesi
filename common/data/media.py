from __future__ import annotations

import ast
import dataclasses
from abc import abstractmethod, ABC
from typing import Any, Optional

from common.data.utils import sanitize_for_ast


@dataclasses.dataclass
class Media(ABC):
    data: Any
    file_path: Optional[str]
    eid: Optional[str]

    def as_mod_tuple(self) -> tuple[str, Media]:
        return self.modality_code(), self

    @staticmethod
    @abstractmethod
    def modality_code() -> str:
        pass

    @classmethod
    def restore_from_dict(cls, data: dict, base_path: str = None) -> Optional[Media]:
        fields = [f.name for f in dataclasses.fields(cls)]
        prefix = cls.modality_code() + "_"

        # An example would be when text is missing.
        if prefix + "metadata" not in data:
            return None

        # Expand my metadata (related to cls)
        file_path = (base_path if base_path is not None else "") + "/" + data[prefix + "file_path"]
        local_data = data | ast.literal_eval(data[prefix + "metadata"]) | {"file_path": file_path}

        restored = {
            attr: (local_data[attr] if attr in local_data else None) for attr in fields
        }

        return cls(**restored)

    @abstractmethod
    def export(self, base_path: str, output_path_to_relative: str = None):
        pass

    def to_dict(self, metadata_only: bool = False) -> dict:
        attrs = [f.name for f in dataclasses.fields(self)]
        metadata = {attr: getattr(self, attr) for attr in attrs if attr != "data" and attr != "file_path"}
        metadata = sanitize_for_ast(metadata)

        if metadata_only:
            return metadata

        return {f"{self.modality_code()}_file_path": self.file_path, f"{self.modality_code()}_metadata": metadata}

    # TODO Deploy questa soluzione più generica che meglio. Posso plug and play cose facilmente così
    #       todo vedi con serialization
    @classmethod
    def restore_from_dict_new(cls, data: dict, base_path: str = None) -> Optional[Media]:
        fields = [f.name for f in dataclasses.fields(cls)]
        file_path = (base_path if base_path is not None else "") + "/" + data["file_path"]
        local_data = data | {"file_path": file_path}
        restored = {attr: (local_data[attr] if attr in local_data else None) for attr in fields}
        return cls(**restored)

    def to_dict_new(self) -> dict:
        attrs = [f.name for f in dataclasses.fields(self)]
        data = {attr: getattr(self, attr) for attr in attrs if attr != "data"}
        data["classname"] = self.__class__.__name__
        return sanitize_for_ast(data)



from __future__ import annotations

import dataclasses
from abc import abstractmethod, ABC
from typing import Any, Optional

import torch

from main.core_data.utils import sanitize_for_ast


@dataclasses.dataclass
class Media(ABC):
    data: Any
    eid: Optional[str]

    def as_mod_tuple(self) -> tuple[str, Media]:
        return self.modality_code(), self

    @staticmethod
    @abstractmethod
    def modality_code() -> str:
        pass

    @abstractmethod
    def export(self, base_path: str, output_path_to_relative: str = None):
        pass

    def clone(self, substitutions: dict):
        new_object = dataclasses.replace(self)
        for key, value in substitutions.items():
            # Only substitute existing parameters when given.
            if hasattr(self, key):
                new_object.__setattr__(key, value)

        if isinstance(self.data, torch.Tensor):
            new_object.data = self.data.detach().clone()

        return new_object

    @classmethod
    def restore_from_dict(cls, data: dict, base_path: str = None) -> Optional[Media]:
        fields = [f.name for f in dataclasses.fields(cls)]
        file_path = (base_path if base_path is not None else "") + "/" + data["file_path"]
        local_data = data | {"file_path": file_path}
        restored = {attr: (local_data[attr] if attr in local_data else None) for attr in fields}
        return cls(**restored)

    def to_dict(self) -> dict:
        attrs = [f.name for f in dataclasses.fields(self)]
        data = {attr: getattr(self, attr) for attr in attrs if attr != "data"}
        data["classname"] = self.__class__.__name__
        return sanitize_for_ast(data)

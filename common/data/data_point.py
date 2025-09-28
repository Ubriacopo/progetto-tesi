from __future__ import annotations

import ast
from abc import ABC, abstractmethod
from dataclasses import is_dataclass, replace
from typing import Optional

from torch import nn

from common.data import media_types
from common.data.media import Media


class DatasetDataPoint(ABC):
    @abstractmethod
    def to_dict(self) -> dict:
        pass

    @staticmethod
    @abstractmethod
    def from_dict(o: dict) -> DatasetDataPoint:
        pass

    @staticmethod
    @abstractmethod
    def get_identifier() -> str:
        pass


class FlexibleDatasetPoint(DatasetDataPoint):
    def __init__(self, eid: str | int, *modality: tuple[str, dict | Media]):
        self.eid = eid
        for (k, o) in modality:
            self.__setattr__(k, o)

    def clone(self, n_eid: str) -> FlexibleDatasetPoint:
        modalities: list[tuple[str, dict | Media]] = []
        for attr, value in self.__dict__.items():
            if hasattr(value, "clone"):
                # Special case of Media.
                modalities.append((attr, value.clone({"eid": n_eid})))
            elif is_dataclass(value):
                modalities.append((attr, replace(value, eid=n_eid)))
            else:
                modalities.append((attr, value))

        return FlexibleDatasetPoint(n_eid, *modalities)

    def to_dict(self) -> dict:
        o = {self.get_identifier(): self.eid}
        for attr, value in self.__dict__.items():
            if isinstance(value, Media) or hasattr(value, "to_dict_new"):
                # Custom dict logic. Should add classname to it for restore?
                value = value.to_dict()

            o |= {attr: value}

        return o

    def __getitem__(self, item: str):
        return getattr(self, item)

    def __setitem__(self, key, value):
        self.__setattr__(key, value)

    @staticmethod
    def get_identifier() -> str:
        return "eid"

    @staticmethod
    def from_dict(o: dict, base_path: str = None) -> FlexibleDatasetPoint:
        objects = []
        for attr, value in o.items():
            # Only exception we can handle inside the DatasetPoint
            if attr == FlexibleDatasetPoint.get_identifier():
                continue

            if isinstance(value, str):
                # Turn into a dict if it is one
                value = ast.literal_eval(value)

            assert isinstance(value, dict)
            # Path of restoring the object of type else it is just dict
            if "classname" in value:
                media_type: Media = getattr(media_types, value["classname"])
                objects.append((attr, media_type.restore_from_dict(value)))
            else:
                objects.append((attr, value))
        # Flexible to any new structure
        return FlexibleDatasetPoint(o["eid"], *objects)

    def export(self, base_path: str = None, relative_path: str = None, *exceptions: str, only: str = None):
        # Only one execution branch
        if only is not None and hasattr(self, only):
            attr = self.__getattribute__(only)
            if isinstance(attr, Media) or hasattr(attr, "export"):
                attr.export(base_path, relative_path)
                return

        # All except the exceptions
        for attr, value in self.__dict__.items():
            if not isinstance(value, Media) and not hasattr(value, "export"):
                continue

            # Ignore the currently processed element.
            if attr in exceptions:
                continue

            value.export(base_path, relative_path)


class FlexibleDatasetTransformWrapper:
    def __init__(self, name: str, *transforms: tuple[str, nn.Module],
                 # If nested are to expand and what keys we want to expand
                 expand_nested: bool = False, nested_keys: list[str] = None,
                 shared_pipeline: nn.Module = None):
        """
        A custom definable transform wrapper that works on existing modalities contained in any AgnosticDatasetPoint
        :param name: Name of the transform to identify the process.
        :param transforms: Torch transforms to ensure max compatibility. (For torch-script)
        :param expand_nested: If nested dictionaries have to be flattened. Works with nested keys.
        :param nested_keys: What subkeys are to expand to upper level.
        :param shared_pipeline: Shared pipeline to all modalities in the FlexibleDatasetPoint.
                                It gives more freedom but I'd avoid it. Ignored at the moment. TODO: Implement
        """
        self.name: str = name
        self.expand_nested: bool = expand_nested
        self.nested_keys: Optional[list[str]] = nested_keys

        # This allows
        self.shared_pipeline: nn.Module = shared_pipeline

        for (k, o) in transforms:
            self.__setattr__(k, o)

    def __getitem__(self, item: str):
        return getattr(self, item)

    def is_defined(self, item: str):
        return item in self.__dict__

    def call(self, x: FlexibleDatasetPoint, keep_type: bool = False):
        y = {} if not keep_type else x
        for key, value in x.__dict__.items():
            if self.is_defined(key):
                # Call each transform that maps to x definition
                y[key] = self[key](value)
                # If the generation implies expanding keys (Example would be audio that generates text as well)
                # they are iterated and expanded. I can only work if result of pipeline is dict like
                if self.expand_nested and isinstance(y[key], dict):
                    for expand_key in self.nested_keys:
                        if expand_key in y[key]:
                            y[expand_key] = y[key].pop(expand_key)

        return y

import dataclasses
from abc import ABC
from typing import Optional

import mne

from core_data.media.media import Media


@dataclasses.dataclass
class Signal(Media, ABC):
    fs: int
    interval: Optional[tuple[int, int]] = dataclasses.field(default=None, kw_only=True)

    def clone(self, substitutions: dict):
        new_object = super().clone(substitutions)
        if isinstance(self.data, mne.io.RawArray):
            new_object.data = self.data.copy()
        return new_object

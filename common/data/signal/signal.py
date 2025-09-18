import dataclasses
from abc import ABC
from typing import Optional

from common.data.media import Media

@dataclasses.dataclass
class Signal(Media, ABC):
    fs: int
    interval: Optional[tuple[int, int]] = dataclasses.field(default=None, kw_only=True)

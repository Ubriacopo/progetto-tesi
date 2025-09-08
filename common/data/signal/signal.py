from abc import ABC
from typing import Optional

from common.data.media import Media


class Signal(Media, ABC):
    fs: float
    interval: Optional[tuple[int, int]] = None

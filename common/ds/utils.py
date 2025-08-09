from collections import OrderedDict


class BoundedMap(OrderedDict):
    def __init__(self, max_length: int, lru: bool = False):
        super().__init__()
        self.max_length, self.lru = max_length, lru

    def __getitem__(self, key):
        v = super().__getitem__(key)
        if self.lru:
            self.move_to_end(key)  # mark as most-recent
        return v

    def __setitem__(self, key, value):
        if key in self and self.lru:
            self.move_to_end(key)
        super().__setitem__(key, value)
        if len(self) > self.max_length:
            # Evict the oldest entry
            self.popitem(last=False)

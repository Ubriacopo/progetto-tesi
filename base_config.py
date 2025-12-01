import os


class BaseConfig:
    SUPPRESS_TIMED: bool = os.getenv('SUPPRESS_TIMED', False)

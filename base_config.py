import os


class BaseConfig:
    SUPPRESS_TIMED: bool = os.getenv('SUPPRESS_TIMED', True)
    SUPPRESS_ENTER_LEAVE_LOG: bool = os.getenv('SUPPRESS_ENTER_LEAVE_LOG', True)

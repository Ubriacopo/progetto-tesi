import os


class BaseConfig:
    SUPPRESS_TIMED: bool = os.getenv('SUPPRESS_TIMED', False)
    SUPPRESS_ENTER_LEAVE_LOG: bool = os.getenv('SUPPRESS_ENTER_LEAVE_LOG', False)

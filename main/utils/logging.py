import logging

import colorlog

def make_logger(name: str):
    return logging.getLogger(name)

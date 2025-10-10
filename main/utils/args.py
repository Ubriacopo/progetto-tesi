import argparse
import inspect
import json


def work_with_config_file(default: str = None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=str, help="Path to JSON config", default=default
    )

    args = parser.parse_args()
    with open(args.config) as f:
        cfg: dict = json.load(f)

    return cfg


def safe_call(func):
    def wrapper(**kwargs):
        sig = inspect.signature(func)
        filtered = {k: v for k, v in kwargs.items() if k in sig.parameters}
        return func(**filtered)

    return wrapper

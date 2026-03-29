# Come li salvo?
from pathlib import Path

import torch
from tensordict import TensorDict

from main.model.neegavi.model import EegInterAviModel


class EegaviPersistentForward:
    def __init__(self, model: EegInterAviModel, output_path: str):
        self.model: EegInterAviModel = model
        self.output_path = Path(output_path)
        self.output_path.mkdir(parents=True, exist_ok=True)

    def process(self, sample: TensorDict, keep_keys: list[str] = ("eeg", "assessment")) -> TensorDict:
        return_object = {}
        for key in keep_keys:
            if key in sample:
                return_object[key] = sample[key]
            else:
                # log warn
                print("Key missing from sample")

        with torch.inference_mode():
            res = self.model(sample, use_kd=False)

        return_object["fused"] = res.cls
        return TensorDict.from_dict(return_object)

    def store(self, sample: TensorDict):
        # Store it.
        sample.memmap(str(self.output_path), copy_existing=True)

    def __call__(self, sample: TensorDict):
        obj = self.process(sample)
        self.store(obj)
        return obj

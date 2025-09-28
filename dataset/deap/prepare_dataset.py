import argparse
import json

from core_data import sampler
from dataset.deap.loader import DeapPointsLoader
from dataset.deap.preprocessing import deap_interleaved_preprocessor
from core_data.media.eeg.config import EegTargetConfig
from core_data.media.video import VidTargetConfig

parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, required=True, help="Path to JSON config")
args = parser.parse_args()

with open(args.config) as f:
    cfg: dict = json.load(f)

kwargs = {}

if "vid_config" in cfg:
    vid_config = VidTargetConfig()
    for k, v in cfg["vid_config"].items():
        setattr(vid_config, k, v)
    kwargs["vid_config"] = vid_config

if "eeg_config" in cfg:
    eeg_config = EegTargetConfig(cfg["eeg_config"]["cbramod_weights_path"])
    for k, v in cfg["eeg_config"].items():
        setattr(eeg_config, k, v)
    kwargs["eeg_config"] = eeg_config

if "segmenter" in cfg:
    segmenter_type = cfg["segmenter"]["type"]
    segmenter_type = getattr(sampler, segmenter_type)
    segmenter = segmenter_type(**cfg["segmenter"]["kwargs"])
    kwargs["segmenter"] = segmenter

print("Deap process starting")
processor = deap_interleaved_preprocessor(
    cfg["output_max_length"], cfg["output_path"],
    **kwargs
)

processor.run(DeapPointsLoader(cfg["data_path"]))

import argparse
import json

from common.data import sampler
from common.data.amigos.config import AmigosConfig
from common.data.amigos.loader import AmigosPointsLoader
from common.data.amigos.preprocessing import amigos_interleaved_preprocessor
from common.data.audio.config import AudTargetConfig
from common.data.ecg.config import EcgTargetConfig
from common.data.eeg.config import EegTargetConfig
from common.data.text.config import TxtTargetConfig
from common.data.video.config import VidTargetConfig

parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, help="Path to JSON config", default="./interleaved_prepare_default.json")
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

if "aud_config" in cfg:
    aud_config = AudTargetConfig()
    for k, v in cfg["aud_config"].items():
        setattr(aud_config, k, v)
    kwargs["aud_config"] = aud_config

if "ecg_config" in cfg:
    ecg_config = EcgTargetConfig(prepare=AmigosConfig.prepare_ecg)
    for k, v in cfg["ecg_config"].items():
        setattr(ecg_config, k, v)
    kwargs["ecg_config"] = ecg_config

if "txt_config" in cfg:
    txt_config = TxtTargetConfig(cfg["txt_config"]["registry_store_path"])
    for k, v in cfg["txt_config"].items():
        setattr(txt_config, k, v)
    kwargs["txt_config"] = txt_config

if "segmenter" in cfg:
    segmenter_type = cfg["segmenter"]["type"]
    segmenter_type = getattr(sampler, segmenter_type)
    segmenter = segmenter_type(**cfg["segmenter"]["kwargs"])
    kwargs["segmenter"] = segmenter

print("AMIGOS process starting")
processor = amigos_interleaved_preprocessor(
    cfg["output_max_length"], cfg["output_path"],
    **kwargs
)

processor.run(AmigosPointsLoader(cfg["data_path"]))

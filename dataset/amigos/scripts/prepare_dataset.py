import argparse
import json

from core_data import sampler
from dataset.amigos.config import AmigosConfig
from dataset.amigos.loader import AmigosPointsLoader
from dataset.amigos.preprocessing import amigos_interleaved_preprocessor
from core_data.media.audio import AudTargetConfig
from core_data.media.ecg import EcgTargetConfig
from core_data.media.eeg.config import EegTargetConfig
from core_data.media.text import TxtTargetConfig
from core_data.media.video import VidTargetConfig
from dataset.amigos import preprocessing

parser = argparse.ArgumentParser()
parser.add_argument(
    "--config", type=str, help="Path to JSON config", default="./prepare_dataset_interleaved_default.json"
)
args = parser.parse_args()

with open(args.config) as f:
    cfg: dict = json.load(f)

kwargs = {}

base_path = cfg["base_path"]
print(f"Working for dir: {base_path}")
print(f"Working with file : {args.config}")
if "vid_config" in cfg:
    vid_config = VidTargetConfig()
    for k, v in cfg["vid_config"].items():
        setattr(vid_config, k, v)
    kwargs["vid_config"] = vid_config

if "eeg_config" in cfg:
    eeg_config = EegTargetConfig(base_path + cfg["eeg_config"]["cbramod_weights_path"])
    for k, v in cfg["eeg_config"].items():
        if k != "cbramod_weights_path":
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
    txt_config = TxtTargetConfig()
    for k, v in cfg["txt_config"].items():
        if k != "registry_store_path":
            setattr(txt_config, k, v)
    kwargs["txt_config"] = txt_config

if "output_max_length" in cfg:
    kwargs["output_max_length"] = cfg["output_max_length"]

if "output_path" in cfg:
    kwargs["output_path"] = base_path + cfg["output_path"]

if "extraction_data_folder" in cfg:
    kwargs["extraction_data_folder"] = base_path + cfg["extraction_data_folder"]

print("AMIGOS process starting")
processor = getattr(preprocessing, cfg["processor"])(**kwargs)
processor.run(AmigosPointsLoader(base_path + cfg["data_path"]), workers=1)

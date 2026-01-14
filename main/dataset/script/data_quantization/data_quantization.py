import dataclasses
from pathlib import Path

import hydra
import pandas as pd
import tensordict
import torch
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf

from main.dataset.quantization import Float16ToInt8Quantization
from main.utils.logging import make_logger


@dataclasses.dataclass
class Config:
    shard_size_bytes: float
    export_path: str
    quantization_keys: list[str]
    ds_path: str


cs = ConfigStore.instance()
OmegaConf.register_new_resolver("capitalize", lambda s: s.capitalize())
OmegaConf.register_new_resolver("uppercase", lambda s: s.upper())


# setups
# MANHOB
# ds_path=/mnt/turing-datasets/EEGAVI/MAHNOB/interleaved/
# export_path=/localssd/EEGAVI/MANHOB/interleaved_quantized/



# todo little refactor
@hydra.main(version_base=None, config_name="base", config_path="config")
def main(cfg: Config):
    """
    Since the data is erroneously stored in fp32 (should be fp16) we change it.
    By checking some of the data going from fp16 -> int8 + scales loss of info is almost 0.
    We so quantize to further reduce the space we use.

    For performance reasons I also should be changing the sharding.
    I want bigger shards. (Up to 3-5 GB each)
    """
    # allow extra keys only on txt_config
    logger = make_logger("prepare_ds_pre_extracted")
    logger.info(OmegaConf.to_yaml(cfg))
    OmegaConf.set_struct(cfg, False)
    OmegaConf.to_container(cfg, resolve=True)

    quantizer = Float16ToInt8Quantization()
    spec = pd.read_csv(cfg.ds_path + "spec.csv")
    # Now we iterate and prepare chunks
    old_eids = []
    stack = []
    current_stack_size = 0
    Path.mkdir(Path(cfg.export_path), exist_ok=True, parents=True)

    existing_df = pd.DataFrame(columns=["eid", "sharded_eid", "sharded_index", "index", "segment"])
    if Path(cfg.export_path + "spec.csv").exists():
        existing_df = pd.read_csv(cfg.export_path + "spec.csv")

    ds_path = Path(cfg.ds_path)
    sharded_eid: int = 0 if len(existing_df["sharded_eid"]) == 0 else max(existing_df["sharded_eid"])
    for folder in ds_path.iterdir():
        if not folder.is_dir() or (existing_df["eid"] == int(folder.stem)).any():
            # We hit spec.csv again or the procedure started but was not finished
            logger.info(f"Skipping {folder.stem} as it was already quantized before.")
            continue

        td = tensordict.load_memmap(folder)
        # We add a new metadata key. The old experiment seems wrong?
        # It is not wrong, but it references old data structure (original ds). We are working on a new one.
        td["meta"]["eid"] = torch.tensor(int(folder.stem), dtype=torch.int).repeat(td["meta"].batch_size[0])
        # Remove assessment if exists. We decided to discard it as it brings little info
        if "assessment" in td:
            del td["assessment"]

        for quantize_key in cfg.quantization_keys:
            if quantize_key in td:
                data = td[quantize_key]["data"]
                data, scales = quantizer.quantize(data)
                quantizer.check_loss(td[quantize_key]["data"], data, scales)
                td[quantize_key]["data"] = data
                td[quantize_key]["scales"] = scales

        td_size = sum(v.numel() * v.element_size() for v in td.values(True, True) if hasattr(v, "numel"))
        if td_size + current_stack_size > cfg.shard_size_bytes:
            # Read the spec csv and cat the rows add new col for sharded aggregation key
            df = pd.DataFrame()
            for eid in old_eids:
                item = int(eid)
                df = pd.concat((df, spec[spec["eid"] == item]))
            # Now we can look up from this. We need to be able to see the relative index now
            df["sharded_eid"] = sharded_eid
            df["sharded_index"] = range(len(df))
            # Change index so that it now makes sense
            new_td = tensordict.cat(stack, dim=0)
            new_td.save(cfg.export_path + str(sharded_eid))

            # todo add new indexing
            existing_df = pd.concat((existing_df, df))
            existing_df.to_csv(cfg.export_path + "spec.csv", index=False)
            # Remove from fs the tds we had.

            stack = []
            old_eids = []
            current_stack_size = 0
            sharded_eid += 1

        # Add to stack if the size of the stack allows it
        current_stack_size += td_size
        stack.append(td)
        old_eids.append(td["meta"]["eid"][0])

    df = pd.DataFrame()
    for eid in old_eids:
        item = int(eid)
        df = pd.concat((df, spec[spec["eid"] == item]))
    # Now we can look up from this. We need to be able to see the relative index now
    df["sharded_eid"] = sharded_eid
    df["sharded_index"] = range(len(df))
    # Change index so that it now makes sense
    new_td = tensordict.cat(stack, dim=0)
    new_td.save(cfg.export_path + str(sharded_eid))

    # todo add new indexing
    existing_df = pd.concat((existing_df, df))
    existing_df.to_csv(cfg.export_path + "spec.csv", index=False)
    # Remove from fs the tds we had.
    logger.info("Done doing our stuff!")


if __name__ == "__main__":
    main()

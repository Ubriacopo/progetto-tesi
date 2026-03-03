import hydra
import torch
import torchinfo
from lightning.pytorch.profilers import SimpleProfiler

from main.model.neegavi.helpers import default_trainer, build_easy_eegavi_module
from main.model.script.hydra_beans import KdConfig
from main.utils.logging import make_logger


@hydra.main(config_path="../../../conf", config_name="train")
def main(cfg: KdConfig):
    # cfg = OmegaConf.to_container(cfg, resolve=True)
    torch.manual_seed(cfg.seed)  # Reproducibility
    logger = make_logger("hydra-main.train")
    module = build_easy_eegavi_module(cfg)
    torchinfo.summary(module)

    module.datamodule.setup("")
    train_batches = module.datamodule.size("train")

    model_name = "TODO"
    accumulate_target = 128

    profiling = False
    profiler = SimpleProfiler() if profiling else None
    trainer = default_trainer(
        epochs=20,
        model_name=model_name,
        profiler=profiler,
        limit_train_batches=train_batches,  # TODO Calculate from batch size
        monitor_key="val_global/fused/bidirectional/mrr_mean",
        accumulate_grad_batches=int(accumulate_target / cfg.trainer.batch_size)
    )

    trainer.fit(module, datamodule=module.datamodule, ckpt_path=cfg.trainer.ckpt_path)
    if profiler is not None:
        logger.info(profiler.summary())


if __name__ == "__main__":
    main()

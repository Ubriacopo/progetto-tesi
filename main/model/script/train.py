import hydra
import lightning
import torchinfo
from lightning.pytorch.profilers import SimpleProfiler
from omegaconf import OmegaConf

from main.model.neegavi.helpers import default_trainer, build_easy_eegavi_module
from main.model.script.hydra_beans import KdConfig
from main.utils.logging import make_logger


@hydra.main(config_path="../../../conf", config_name="train")
def main(cfg: KdConfig):
    lightning.seed_everything(cfg.seed, workers=True)
    logger = make_logger("hydra-main.train")
    logger.info(OmegaConf.to_yaml(cfg))

    module = build_easy_eegavi_module(cfg)
    torchinfo.summary(module)

    module.datamodule.setup("")
    train_batches = module.datamodule.size("train")

    model_name = "EEGAVI"
    profiling = False
    profiler = SimpleProfiler() if profiling else None

    trainer = default_trainer(epochs=40, model_name=model_name, profiler=profiler, limit_train_batches=train_batches,
                              # TODO mettere accumulation e mini batch size in config
                              monitor_key="val/fused/mrr_mean", accumulate_grad_batches=1)

    trainer.fit(module, datamodule=module.datamodule, ckpt_path=cfg.trainer.ckpt_path)
    if profiler is not None:
        logger.info(profiler.summary())


if __name__ == "__main__":
    main()

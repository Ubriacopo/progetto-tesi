from __future__ import annotations

import copy
import dataclasses
import gc
from abc import ABC

import lightning
import optuna
import torch
from lightning.pytorch.callbacks import RichProgressBar, EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger

from main.model.neegavi.helpers import build_easy_eegavi_module
from main.model.script.hydra_beans import TuningKdConfig
from main.utils.logging import make_logger


@dataclasses.dataclass
class Definition[T](ABC):
    pass


@dataclasses.dataclass
class ChoiceDefinition[T](Definition[T]):
    values: list[T]


@dataclasses.dataclass
class StepDefinition[T](Definition[T]):
    low: T
    high: T
    step: T | None
    is_log: bool

    def __post_init__(self):
        # reject bool explicitly (since bool is a subclass of int)
        if isinstance(self.low, bool) or isinstance(self.high, bool) or isinstance(self.step, bool):
            raise TypeError("StepDefinition does not accept bool.")


@dataclasses.dataclass
class TuningSearchSpace:
    lr: Definition[float]
    batch_size: Definition[int]
    attn_layers: Definition[int]
    beta: Definition[float]

    def suggest(self, key: str, trial: optuna.Trial):
        if not hasattr(self, key):
            raise IndexError("Invalid trial key: {}".format(key))

        o = self.__getattribute__(key)
        if isinstance(o, ChoiceDefinition):
            return trial.suggest_categorical(key, o.values)

        if isinstance(o, StepDefinition):
            if isinstance(o.low, int):
                step = 1 if o.step is None else o.step
                return trial.suggest_int(key, low=o.low, high=o.high, step=step, log=o.is_log)
            return trial.suggest_float(key, low=o.low, high=o.high, step=o.step, log=o.is_log)

        raise TypeError("Invalid set object type: {}".format(type(o)))

    @staticmethod
    def default() -> TuningSearchSpace:
        return TuningSearchSpace(
            # CLIP-like contrastive: 5e-5 – 5e-4
            # Transformer encoders: 1e-5 – 3e-4
            # Thus: [3e-5, 1e-4, 3e-4, 1e-3]
            lr=ChoiceDefinition(values=[3e-5, 1e-4, 3e-4, 1e-3]),
            batch_size=ChoiceDefinition(values=[32, 64, 128]),
            attn_layers=ChoiceDefinition(values=[2, 4, 6]),
            beta=ChoiceDefinition(values=[0.25, 0.5, 1, 2.0]),
        )

    @staticmethod
    def from_choices(
            lr: list = (3e-5, 1e-4, 3e-4, 1e-3),
            batch_size: list = (32, 64, 128),
            attn_layers: list = (2, 4, 6),
            beta: list = (0.25, 0.5, 1, 2.0)
    ):
        return TuningSearchSpace(
            lr=ChoiceDefinition(values=lr),
            batch_size=ChoiceDefinition(values=batch_size),
            attn_layers=ChoiceDefinition(values=attn_layers),
            beta=ChoiceDefinition(values=beta),
        )


def make_trial_config(trial: optuna.Trial, cfg: TuningKdConfig, logger, search_space: TuningSearchSpace):
    custom_config = copy.deepcopy(cfg)
    custom_config.trainer.lr = search_space.suggest("lr", trial)
    batch_size = search_space.suggest("batch_size", trial)

    accumulation = 1
    # Memory trick for local
    attention_layers = search_space.suggest("attn_layers", trial)
    custom_config.model.factory.args.attention_config = attention_layers
    # todo piu parametri
    if cfg.use_trick and cfg.trick_batch_size <= batch_size and (attention_layers > 4 or batch_size >= 64):
        logger.info(f"Using accumulation trick for attention layers: {attention_layers}")
        # For me this is enough. No need to engineer a good solution.
        factor = 2 if batch_size == 64 or batch_size == 32 else 8
        batch_size = int(batch_size / factor)
        accumulation = factor  # We have to double the accumulation
        logger.info("Accumulation=" + str(accumulation))

    # We store 16 with trick but actual batch is 32 thus the search_space tracks correctly.
    custom_config.trainer.batch_size = batch_size
    custom_config.trainer.kd_loss_weight = search_space.suggest("beta", trial)

    return custom_config, accumulation


def find_duplicates(trial: optuna.Trial, logger=make_logger("tuning.find_duplicates")) -> float | None:
    for t in trial.study.trials:
        if t.state != optuna.trial.TrialState.COMPLETE:
            continue

        if t.params == trial.params:
            logger.info(f"Found duplicate trial, we return same value: {t.params}")
            return t.value

    return None


def final_objective(trial: optuna.Trial, cfg: TuningKdConfig, search_space: TuningSearchSpace,
                    avoid_duplicates: bool = True, epochs: int = 50, patience: int = 8, seed_override: int = None):
    module, trainer = None, None
    try:
        logger = make_logger("final_objective")
        lightning.seed_everything(cfg.seed if seed_override is None else seed_override, workers=True)
        custom_config, accumulation = make_trial_config(trial, cfg, logger, search_space)

        if avoid_duplicates:
            duplicate = find_duplicates(trial, logger)
            if duplicate is not None:
                return duplicate

        logger.info(f"Running trial on params: {trial.params}")
        module = build_easy_eegavi_module(custom_config)
        module.datamodule.setup("")

        train_batches = module.datamodule.size("train")
        monitor_key = "val/fused/mrr_mean"

        early_stopping = EarlyStopping(
            monitor=monitor_key, min_delta=0.002, patience=patience, mode="max", verbose=True
        )

        trainer = lightning.Trainer(
            accelerator="gpu", devices=1,
            logger=TensorBoardLogger("tb_logs", name=trial.study.study_name, version=str(trial.number)),
            callbacks=[
                RichProgressBar(),
                early_stopping,
                ModelCheckpoint(
                    dirpath="checkpoints",
                    filename="epoch{epoch}-step{step}",
                    save_top_k=1,
                    save_last=True,
                    monitor=monitor_key,
                    mode="max"
                )
            ],
            precision="16-mixed",
            max_epochs=epochs,
            max_steps=int(train_batches * epochs),
            limit_train_batches=train_batches,
            val_check_interval=1.0,
            accumulate_grad_batches=accumulation,
            log_every_n_steps=20
        )

        trainer.fit(module, datamodule=module.datamodule)
        score = early_stopping.best_score

        if score is None:
            raise KeyError(f"Metric '{monitor_key}' was not tracked.")

        return float(score.item())

    finally:
        if trainer is not None:
            del trainer
        if module is not None:
            del module

        gc.collect()
        torch.cuda.empty_cache()


def refine_objective(trial: optuna.Trial, cfg: TuningKdConfig, search_space: TuningSearchSpace,
                     avoid_duplicates: bool = True, epochs: int = 15, patience: int = 5, seed_override: int = None):
    module, trainer = None, None
    try:

        logger = make_logger("refine_objective")
        lightning.seed_everything(cfg.seed if seed_override is None else seed_override, workers=True)
        custom_config, accumulation = make_trial_config(trial, cfg, logger, search_space)

        if avoid_duplicates:
            duplicate = find_duplicates(trial, logger)
            if duplicate is not None:
                return duplicate

        logger.info(f"Running trial on params: {trial.params}")

        module = build_easy_eegavi_module(custom_config, 0.3)
        module.datamodule.setup("")

        train_batches = module.datamodule.size("train")
        monitor_key = "val/fused/mrr_mean"

        early_stopping = EarlyStopping(
            monitor=monitor_key, min_delta=0.002, patience=patience, mode="max", verbose=True
        )

        trainer = lightning.Trainer(
            accelerator="gpu", devices=1,
            logger=TensorBoardLogger("tb_logs", name=trial.study.study_name, version=str(trial.number)),
            callbacks=[RichProgressBar(), early_stopping, ],
            precision="16-mixed",
            max_epochs=epochs,
            max_steps=int(train_batches * epochs),
            limit_train_batches=train_batches,
            val_check_interval=1.0,
            accumulate_grad_batches=accumulation,
            log_every_n_steps=20
        )

        trainer.fit(module, datamodule=module.datamodule)
        score = early_stopping.best_score
        if score is None:
            raise KeyError(f"Metric '{monitor_key}' was not tracked.")

        return float(score.item())

    finally:
        if trainer is not None:
            del trainer
        if module is not None:
            del module

        gc.collect()
        torch.cuda.empty_cache()


# Explorative objective
def objective(trial: optuna.Trial, cfg: TuningKdConfig, search_space: TuningSearchSpace,
              avoid_duplicates: bool = True,
              # Should be 1277, but we train on 30% of the data
              reference_train_batches: int = 384, reference_epochs: int = 5) -> float:
    """

    :param avoid_duplicates:
    :param reference_train_batches:
    :param trial:
    :param cfg:
    :param search_space:
    :param reference_epochs: Max number of training epochs (or steps?)
    :return:
    """
    module, trainer = None, None
    try:
        logger = make_logger("objective")
        lightning.seed_everything(cfg.seed, workers=True)
        custom_config, accumulation = make_trial_config(trial, cfg, logger, search_space)

        if avoid_duplicates:
            duplicate = find_duplicates(trial, logger)
            if duplicate is not None:
                return duplicate

        logger.info(f"Running trial on params: {trial.params}")
        module = build_easy_eegavi_module(custom_config, 0.3)

        module.datamodule.setup("")
        train_batches = module.datamodule.size("train")
        # Reference acc = 1
        steps = reference_epochs * reference_train_batches
        batch_size = custom_config.trainer.batch_size
        logger.info(
            f"Steps: {steps}, train_batches={train_batches}, micro_bs={batch_size}, acc={accumulation}, effective_bs={batch_size * accumulation}"
        )

        monitor_key = "test/fused/mrr_mean"
        trainer = lightning.Trainer(
            accelerator="gpu",
            logger=TensorBoardLogger("tb_logs", name=trial.study.study_name, version=str(trial.number)),
            devices=1,
            callbacks=[RichProgressBar(), ],
            precision="16-mixed",  # P6000 has no tensor cores
            max_steps=steps,
            log_every_n_steps=20,  # Plot every 1%
            accumulate_grad_batches=accumulation,
            num_sanity_val_steps=0,
            limit_val_batches=0,
        )

        trainer.fit(module, datamodule=module.datamodule)
        results = trainer.test(module, dataloaders=module.datamodule.val_dataloader())

        score = results[0].get(monitor_key)
        if score is None:
            raise KeyError(f"Metric '{monitor_key}' not found in validation results: {results[0].keys()}")
        return float(score)

    finally:
        if trainer is not None:
            del trainer

        if module is not None:
            del module

        gc.collect()
        torch.cuda.empty_cache()

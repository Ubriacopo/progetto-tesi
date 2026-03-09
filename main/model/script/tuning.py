from functools import partial

import hydra
import optuna

from main.model.neegavi.hp_tuning.tuning import objective, TuningSearchSpace
from main.model.script.hydra_beans import KdConfig


class TuningKdConfig(KdConfig):
    search_space: dict


@hydra.main(config_path="../../../conf", config_name="hp_tuning")
def main(cfg: TuningKdConfig):
    search_space = TuningSearchSpace.from_choices(**cfg.search_space)
    study = optuna.create_study(direction="maximize")
    obj = partial(objective, cfg=cfg, search_space=search_space)
    # Optimize the space
    study.optimize(obj, n_trials=10)  # number of iterations


if __name__ == "__main__":
    main()

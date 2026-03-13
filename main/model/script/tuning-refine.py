from functools import partial

import hydra
import optuna

from main.model.neegavi.hp_tuning.tuning import TuningSearchSpace, refine_objective
from main.model.script.hydra_beans import TuningKdConfig


@hydra.main(config_path="../../../conf", config_name="tuning-refine")
def main(cfg: TuningKdConfig):
    search_space = TuningSearchSpace.from_choices(**cfg.search_space)
    study = optuna.create_study(
        direction="maximize",
        study_name="eegavi-hp-refine",
        storage="sqlite:///../optuna.db",
        load_if_exists=True,
    )
    obj = partial(refine_objective, cfg=cfg, search_space=search_space)
    if cfg.watch_configurations is None or len(cfg.watch_configurations) == 0:
        raise ValueError("You must specify at least one watch configuration")

    for config in cfg.watch_configurations:
        study.enqueue_trial(dict(config))

    # Optimize the space
    study.optimize(
        obj,
        n_trials=len(cfg.watch_configurations),
        gc_after_trial=True,
        show_progress_bar=True
    )  # number of iterations

    print("Best trial:")
    trial = study.best_trial
    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))


if __name__ == "__main__":
    main()

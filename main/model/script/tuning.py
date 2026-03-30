from functools import partial

import hydra
import optuna
from optuna.samplers import TPESampler, GridSampler

from main.model.neegavi.hp_tuning.tuning import objective, TuningSearchSpace, refine_objective
from main.model.script.hydra_beans import TuningKdConfig


@hydra.main(config_path="../../../conf", config_name="hp_tuning")
def main(cfg: TuningKdConfig):
    search_space = TuningSearchSpace.from_choices(**cfg.search_space)
    study = optuna.create_study(
        # Reproducibility
        sampler=GridSampler(seed=cfg.seed, search_space=cfg.search_space),
        direction="maximize",
        study_name=cfg.study_name,
        storage="sqlite:///../optuna-remote.db",
        load_if_exists=True,
    )

    obj = partial(refine_objective, cfg=cfg, search_space=search_space, epochs=4)
    # Optimize the space
    study.optimize(
        obj,
        n_trials=120,
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

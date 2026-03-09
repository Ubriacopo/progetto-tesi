from functools import partial

import hydra
import optuna

from main.model.neegavi.hp_tuning.tuning import objective


@hydra.main(config_path="../../../conf", config_name="hp_tuning")
def main(cfg):
    study = optuna.create_study(direction="maximize")
    obj = partial(objective, param1=param1, param2=param2)
    # Optimize the space
    study.optimize(objective, n_trials=10)  # number of iterations


if __name__ == "__main__":
    main()

from functools import partial

import hydra
import optuna

from main.model.neegavi.tuning import objective


@hydra.main(config_path="../../../conf", config_name="tuning")
def main(cfg):
    study = optuna.create_study(direction="maximize")
    obj = partial(objective, param1=param1, param2=param2)
    # Optimize the space
    study.optimize(objective, n_trials=10)  # number of iterations


if __name__ == "__main__":
    main()

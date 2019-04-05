import os
import logging

from nnattack.variables import auto_var, get_file_name

from main import eps_accuracy
from params import (
    nn_k1,
    nn_k3,
    nn_k5,
    nn_k7,
    robust_nn_k1,
    robust_nn_k3,
    dt_attack,
    robust_rf,
    rf_attack,
    rf500_attack,
    opt_of_nnopt,
    opt_of_rf_attack
)

logging.basicConfig(level=logging.DEBUG)

#auto_var.settings['result_file_dir'] = '/results/'
DEBUG = True if os.environ.get('DEBUG', False) else False

def main():
    experiments = [
        nn_k1, nn_k3, nn_k5,
        nn_k7,
        opt_of_nnopt,
    ]
    experiments += [
        #robust_nn_k1,
        robust_nn_k3
    ]
    experiments += [
        robust_rf,
    ]
    experiments += [
        rf_attack,
        opt_of_rf_attack,
        #rf500_attack,
    ]
    grid_params = []
    for exp in experiments:
        exp_fn, _, grid_param, run_param = exp()
        grid_params += grid_param

    if DEBUG:
        run_param['n_jobs'] = 1
        run_param['allow_failure'] = False
    else:
        run_param['allow_failure'] = True
    auto_var.run_grid_params(exp_fn, grid_params, **run_param)


if __name__ == "__main__":
    main()

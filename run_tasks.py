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
    dt_attack,
    opt_of_nnopt,
)

logging.basicConfig(level=logging.DEBUG)

#auto_var.settings['result_file_dir'] = '/results/'
DEBUG = True if os.environ.get('DEBUG', False) else False

def main():
    experiments = [opt_of_nnopt, nn_k1, robust_nn_k1, nn_k3, nn_k5, nn_k7]
    #experiments = [dt_attack]
    grid_params = []
    for exp in experiments:
        exp_fn, grid_param, run_param = exp()
        grid_params += grid_param

    if DEBUG:
        run_param['n_jobs'] = 1
        run_param['allow_failure'] = False
    else:
        run_param['allow_failure'] = True
    auto_var.run_grid_params(exp_fn, grid_params, **run_param)


if __name__ == "__main__":
    main()

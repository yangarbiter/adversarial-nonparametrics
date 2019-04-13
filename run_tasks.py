import os
import logging

from nnattack.variables import auto_var, get_file_name

from params import (
    #nn_k1,
    #nn_k3,
    #nn_k5,
    #nn_k7,
    #robust_nn_k1, robust_nn_k3, dt_attack,
    #robust_rf,
    #rf_attack,
    #rf500_attack,

    #opt_of_rf_attack,
    #opt_of_nnopt,

    optimality,
    compare_nns,
    nn_k1_robustness,
    nn_k3_robustness,

    rf_robustness,
)
from main import eps_accuracy
from tasks import run_exp

logging.basicConfig(level=logging.DEBUG)

#auto_var.settings['result_file_dir'] = '/results/'
DEBUG = True if os.environ.get('DEBUG', False) else False

def main():
    experiments = [
        #compare_nns,
        nn_k1_robustness,
        nn_k3_robustness,

        optimality,

        rf_robustness,
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
    #auto_var.run_grid_params(delete_file, grid_params, n_jobs=1,
    #                          with_hook=False, allow_failure=False)
    #auto_var.run_grid_params(celery_run, grid_params, n_jobs=1,
    #                         allow_failure=False)

def delete_file(auto_var):
    os.unlink(get_file_name(auto_var) + '.json')

def celery_run(auto_var):
    run_exp.delay(auto_var.var_value)


if __name__ == "__main__":
    main()

import os
import logging

from nnattack.variables import auto_var, get_file_name

from main import eps_accuracy
from params import (
    robustnn_with_opt_attack,
    dt_attack,
    knn1_attack,
    kernel_sub_clf,
    adv_lr,
    adv_sklr,
    adv_sklinsvc,
    adv_nn,
    adv_nn_large,
    adv_nn_k3,
)

logging.basicConfig(level=logging.DEBUG)

#auto_var.settings['result_file_dir'] = '/results/'
DEBUG = True if os.environ.get('DEBUG', False) else False

def main():
    exp_fn, grid_param, run_param = adv_nn_k3()
    if DEBUG:
        run_param['n_jobs'] = 1
    auto_var.run_grid_params(exp_fn, grid_param, **run_param)


if __name__ == "__main__":
    main()

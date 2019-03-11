import os
import logging

from nnattack.variables import auto_var, get_file_name

from main import eps_accuracy
from params import (
    nn_k1
)

logging.basicConfig(level=logging.DEBUG)

#auto_var.settings['result_file_dir'] = '/results/'
DEBUG = True if os.environ.get('DEBUG', False) else False

def main():
    exp_fn, grid_param, run_param = nn_k1()
    if DEBUG:
        run_param['n_jobs'] = 1
    auto_var.run_grid_params(exp_fn, grid_param, **run_param)


if __name__ == "__main__":
    main()

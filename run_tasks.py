import os
import logging
import json

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

    rf_optimality,
    nn_optimality,
    compare_nns,
    nn_k1_robustness,
    nn_k3_robustness,

    dt_robustness,
    rf_robustness,

    nn_k3_optimality_figs,
)
from main import eps_accuracy

logging.basicConfig(level=logging.DEBUG)

#auto_var.settings['result_file_dir'] = '/results/'
DEBUG = True if os.environ.get('DEBUG', False) else False

def main():
    experiments = [
        #compare_nns,

        #nn_k1_robustness,
        #nn_k3_robustness,

        rf_robustness,
        #dt_robustness,

        #rf_optimality,
        #nn_optimality,

        #nn_k3_optimality_figs,
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
    #auto_var.run_grid_params(temp_fix, grid_params, n_jobs=6,
    #                         allow_failure=False, with_hook=False)

def delete_file(auto_var):
    os.unlink(get_file_name(auto_var) + '.json')

def celery_run(auto_var):
    run_exp.delay(auto_var.var_value)

from main import set_random_seed
import numpy as np
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler

def temp_fix(auto_var):
    file_name = get_file_name(auto_var)
    print(file_name)
    if os.path.exists("%s.json" % file_name):
        with open("%s.json" % file_name, "r") as f:
            ret = json.load(f)
        if "tst_score" in ret:
            return
    else:
        return

    random_state = set_random_seed(auto_var)
    ord = auto_var.get_var("ord")

    X, y, eps_list = auto_var.get_var("dataset")
    idxs = np.arange(len(X))
    random_state.shuffle(idxs)
    trnX, tstX, trny, tsty = X[idxs[:-200]], X[idxs[-200:]], y[idxs[:-200]], y[idxs[-200:]]

    scaler = MinMaxScaler()
    trnX = scaler.fit_transform(trnX)
    tstX = scaler.transform(tstX)

    lbl_enc = OneHotEncoder(categories=[np.sort(np.unique(y))], sparse=False)
    #lbl_enc = OneHotEncoder(sparse=False)
    lbl_enc.fit(trny.reshape(-1, 1))

    auto_var.set_intermidiate_variable("lbl_enc", lbl_enc)

    results = []

    auto_var.set_intermidiate_variable("trnX", trnX)
    auto_var.set_intermidiate_variable("trny", trny)

    model_name = auto_var.get_variable_value("model")
    attack_name = auto_var.get_variable_value("attack")
    if 'adv_rf' in model_name:
        pre_model = auto_var.get_var_with_argument('model', model_name[4:])
        pre_model.fit(trnX, trny)
        if 'blackbox' in attack_name:
            auto_var.set_intermidiate_variable("model", pre_model)
    elif 'adv_nn' in model_name and 'blackbox' in attack_name:
        pre_model = auto_var.get_var_with_argument('model', model_name[4:])
        pre_model.fit(trnX, trny)
        auto_var.set_intermidiate_variable("model", pre_model)

    model = auto_var.get_var("model")
    auto_var.set_intermidiate_variable("model", model)
    model.fit(trnX, trny)

    pred = model.predict(tstX)
    ori_tstX, ori_tsty = tstX, tsty # len = 200
    idxs = np.where(pred == tsty)[0]
    random_state.shuffle(idxs)

    augX = None
    if ('adv' in model_name) or ('robustv1' in model_name) or ('robustv2' in model_name):
        assert hasattr(model, 'augX')
        auto_var.set_intermidiate_variable("trnX", model.augX)
        auto_var.set_intermidiate_variable("trny", model.augy)
        augX, augy = model.augX, model.augy

    ret['tst_score'] = (model.predict(ori_tstX) == ori_tsty).mean()
    with open("%s.json" % file_name, "w") as f:
        json.dump(ret, f)

if __name__ == "__main__":
    main()

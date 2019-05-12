import json
import os
import inspect
from functools import partial

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.metrics import pairwise_distances
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)
import keras.backend
#import tensorflow.keras.backend

from nnattack.variables import auto_var


def set_random_seed(auto_var):
    random_seed = auto_var.get_var("random_seed")

    tf.set_random_seed(random_seed)
    np.random.seed(random_seed)

    sess = tf.Session()
    keras.backend.set_session(sess)
    keras.layers.core.K.set_learning_phase(0)
    #tensorflow.keras.backend.set_session(sess)
    sess.run(tf.global_variables_initializer())
    auto_var.set_intermidiate_variable("sess", sess)
    random_state = np.random.RandomState(auto_var.get_var("random_seed"))
    auto_var.set_intermidiate_variable("random_state", random_state)

    return random_state

def baseline_pert(model, trnX, tstX, tsty, perts, ord, constraint=None):
    pred_trn = model.predict(trnX)
    ret = np.copy(perts)
    for i in np.where(model.predict(tstX + perts) == tsty)[0]:
        tX = trnX[pred_trn != tsty[i]]
        if len(tX) == 0:
            continue
        norms = np.linalg.norm(tX - tstX[i], ord=ord, axis=1)
        if constraint is not None and norms.min() > constraint:
            continue
        ret[i] = tX[norms.argmin()] - tstX[i]
    return ret, (model.predict(tstX + perts) == tsty).sum()


def pass_random_state(fn, random_state):
    if 'random_state' in inspect.getfullargspec(fn).args:
        return partial(fn, random_state=random_state)
    return fn

def estimate_model_roubstness(model, X, y, perturbs, eps_list, ord,
        with_baseline=False, trnX=None):
    assert len(eps_list) == len(perturbs), (eps_list, perturbs.shape)
    ret = []
    for i, eps in enumerate(eps_list):
        assert np.all(np.linalg.norm(perturbs[i], axis=1, ord=ord) <= (eps + 1e-6)), (np.linalg.norm(perturbs[i], axis=1, ord=ord), eps)
        if with_baseline:
            assert trnX is not None
            pert, _ = baseline_pert(model, trnX, X, y, perturbs[i], ord, eps)
            temp_tstX = X + pert
        else:
            temp_tstX = X + perturbs[i]

        pred = model.predict(temp_tstX)

        ret.append({
            'eps': eps_list[i],
            'tst_acc': (pred == y).mean().astype(float),
        })
    return ret

def eps_accuracy(auto_var):
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
    lbl_enc.fit(trny.reshape(-1, 1))

    auto_var.set_intermidiate_variable("lbl_enc", lbl_enc)

    ret = {}
    results = []

    auto_var.set_intermidiate_variable("trnX", trnX)
    auto_var.set_intermidiate_variable("trny", trny)

    model_name = auto_var.get_variable_name("model")
    attack_name = auto_var.get_variable_name("attack")
    if 'adv_rf' in model_name:
        pre_model = auto_var.get_var_with_argument('model', model_name[4:])
        pre_model.fit(trnX, trny)
        if 'blackbox' in attack_name:
            auto_var.set_intermidiate_variable("model", pre_model)
    elif 'adv_nn' in model_name and 'blackbox' in attack_name:
        pre_model = auto_var.get_var_with_argument('model', model_name[4:])
        pre_model.fit(trnX, trny)
        auto_var.set_intermidiate_variable("model", pre_model)
    elif 'mlp' in model_name or 'logistic' in model_name:
        auto_var.set_intermidiate_variable("eps_list", eps_list)

    model = auto_var.get_var("model")
    auto_var.set_intermidiate_variable("model", model)
    model.fit(trnX, trny)

    pred = model.predict(tstX)
    ori_tstX, ori_tsty = tstX, tsty # len = 200
    idxs = np.where(pred == tsty)[0]
    random_state.shuffle(idxs)
    tstX, tsty = tstX[idxs[:100]], tsty[idxs[:100]]
    if len(tsty) != 100:
        print("didn't got 100 testing examples, abort.")
        ret['avg_pert'] = {'avg': 0, 'missed_count': 100,}
        ret['tst_score'] = (model.predict(ori_tstX) == ori_tsty).mean()
        if ('adv' in model_name) or ('advPruning' in model_name) or ('robustv2' in model_name):
            ret['aug_len'] = len(model.augX)
        return ret
        #raise ValueError("didn't got 100 testing examples")

    augX = None
    if ('adv' in model_name) or ('advPruning' in model_name) or ('robustv2' in model_name):
        assert hasattr(model, 'augX')
        auto_var.set_intermidiate_variable("trnX", model.augX)
        auto_var.set_intermidiate_variable("trny", model.augy)
        augX, augy = model.augX, model.augy

    if len(tsty) != 100 or \
       (np.unique(auto_var.get_intermidiate_variable('trny'))[0] != None and \
       len(np.unique(auto_var.get_intermidiate_variable('trny'))) == 1):
        tst_perturbs = np.array([np.zeros_like(tstX) for _ in range(len(eps_list))])
        ret['single_label'] = True
        attack_model = None
    else:
        attack_model = auto_var.get_var("attack")
        tst_perturbs = attack_model.perturb(tstX, y=tsty, eps=eps_list)

    ret['tst_score'] = (model.predict(ori_tstX) == ori_tsty).mean()

    #########
    if attack_model is not None and hasattr(attack_model, 'perts'):
        perts = attack_model.perts
    else:
        perts = np.zeros_like(tstX)
        for pert in tst_perturbs:
            pred = model.predict(tstX + pert)
            for i in range(len(pred)):
                if (pred[i] != tsty[i]) and np.linalg.norm(perts[i])==0:
                    perts[i] = pert[i]

    perts = perts.astype(float)
    perts, missed_count = baseline_pert(model, trnX, tstX, tsty, perts, ord)
    if len(np.unique(model.predict(trnX))) > 1:
        assert (model.predict(tstX + perts) == tsty).sum() == 0, model.predict(tstX + perts) == tsty
    else:
        # ignore single label case
        ret['single_label'] = True
    ret['avg_pert'] = {
        'avg': np.linalg.norm(perts, axis=1, ord=ord).mean().astype(float),
        'missed_count': int(missed_count),
    }
    #########

    results = estimate_model_roubstness(
        model, tstX, tsty, tst_perturbs, eps_list, ord, with_baseline=False)
    ret['results'] = results
    baseline_results = estimate_model_roubstness(
        model, tstX, tsty, tst_perturbs, eps_list, ord, with_baseline=True, trnX=trnX)
    ret['baseline_results'] = baseline_results

    ret['trnX_len'] = len(trnX)
    if augX is not None:
        ret['aug_len'] = len(augX)

    print("Average perturbation:", ret['avg_pert']['avg'])
    print("Testing accuracy:", ret['tst_score'])
    return ret

def main():
    auto_var.parse_argparse()
    auto_var.run_single_experiment(eps_accuracy)

if __name__ == '__main__':
    main()

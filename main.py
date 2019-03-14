import json
import os
import inspect
from functools import partial

import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelBinarizer, MinMaxScaler
from sklearn.metrics import pairwise_distances
#import matplotlib.pyplot as plt
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import keras.backend
import tensorflow.keras.backend
#from tensorflow.python.platform import flags
from bistiming import SimpleTimer

from nnattack.variables import auto_var


global random_seed
random_seed = 1126
np.random.seed(random_seed)
#tf.set_random_seed(random_seed)

def set_random_seed(auto_var):
    tf.set_random_seed(auto_var.get_var("random_seed"))
    sess = tf.Session()
    keras.backend.set_session(sess)
    keras.layers.core.K.set_learning_phase(0)
    tensorflow.keras.backend.set_session(sess)
    #sess.run(tf.global_variables_initializer())
    auto_var.set_intermidiate_variable("sess", sess)
    random_state = np.random.RandomState(auto_var.get_var("random_seed"))
    auto_var.set_intermidiate_variable("random_state", random_state)

    return random_state

def pass_random_state(fn, random_state):
    if 'random_state' in inspect.getfullargspec(fn).args:
        return partial(fn, random_state=random_state)
    return fn

#@profile
def eps_accuracy(auto_var):
    random_state = set_random_seed(auto_var)
    ord = auto_var.get_var("ord")

    X, y, eps_list = auto_var.get_var("dataset")
    idxs = np.arange(len(X))
    random_state.shuffle(idxs)
    trnX, tstX, trny, tsty = X[idxs[:-100]], X[idxs[-100:]], y[idxs[:-100]], y[idxs[-100:]]

    scaler = MinMaxScaler()
    trnX = scaler.fit_transform(trnX)
    tstX = scaler.transform(tstX)

    lbl_enc = OneHotEncoder(categories=[np.sort(np.unique(y))], sparse=False)
    #lbl_enc = OneHotEncoder(sparse=False)
    lbl_enc.fit(trny.reshape(-1, 1))

    auto_var.set_intermidiate_variable("lbl_enc", lbl_enc)

    model_name = auto_var.get_variable_value("model")
    if ('adv' in model_name) or ('robust' in model_name):
        ret = []
        ord = auto_var.get_var("ord")
        for i in range(len(eps_list)):
            eps = eps_list[i]

            auto_var.set_intermidiate_variable("trnX", trnX)
            auto_var.set_intermidiate_variable("trny", trny)
            model = auto_var.get_var("model")
            auto_var.set_intermidiate_variable("model", model)
            model.fit(trnX, trny, eps=eps)

            auto_var.set_intermidiate_variable("trnX", model.augX)
            auto_var.set_intermidiate_variable("trny", model.augy)
            augX, augy = model.augX, model.augy

            attack_model = auto_var.get_var("attack")

            tst_perturb = attack_model.perturb(tstX, y=tsty, eps=eps)

            assert np.all(np.linalg.norm(tst_perturb, axis=1, ord=ord) <= (eps + 1e-6)), (np.linalg.norm(tst_perturb, axis=1, ord=ord), eps)
            temp_tstX = tstX + tst_perturb

            tst_pred = model.predict(temp_tstX)
                                                                                    
            ret.append({                                                             
                'eps': eps_list[i],
                'tst_acc': (tst_pred == tsty).mean(),                                
            })                                                                       
            print(ret[-1])

    else:
        augX = None
        auto_var.set_intermidiate_variable("trnX", trnX)
        auto_var.set_intermidiate_variable("trny", trny)
        model = auto_var.get_var("model")
        auto_var.set_intermidiate_variable("model", model)
        model.fit(trnX, trny)

        attack_model = auto_var.get_var("attack")

        tst_perturbs = attack_model.perturb(tstX, y=tsty, eps=eps_list)

        ret = []
        ord = auto_var.get_var("ord")
        for i in range(len(eps_list)):
            eps = eps_list[i]
            assert np.all(np.linalg.norm(tst_perturbs[i], axis=1, ord=ord) <= (eps + 1e-6)), (np.linalg.norm(tst_perturbs[i], axis=1, ord=ord), eps)
            temp_tstX = tstX + tst_perturbs[i]

            tst_pred = model.predict(temp_tstX)
                                                                                    
            ret.append({
                'eps': eps_list[i],
                'tst_acc': (tst_pred == tsty).mean(),
            })
            print(ret[-1])
                                                                                 
    ret = {'results': ret}
    if isinstance(tst_perturbs, list):
        tst_perturbs = tst_perturbs[-1]
    ret['avg_pert'] = np.linalg.norm(tst_perturbs, axis=1, ord=ord).mean()
    ret['trnX_len'] = len(trnX)
    if augX is not None:
        ret['aug_len'] = len(augX)                                               

    print(json.dumps(auto_var.var_value))
    print(json.dumps(ret))
    return ret

def main():
    auto_var.parse_argparse()
    auto_var.run_single_experiment(eps_accuracy)

if __name__ == '__main__':
    main()

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
    eps = auto_var.get_var("eps")
    ord = auto_var.get_var("ord")
    X, y = auto_var.get_var("dataset")
    trnX, tstX, trny, tsty = train_test_split(
        X, y, test_size=auto_var.get_var("test_size"), random_state=random_state)

    scaler = MinMaxScaler()
    trnX = scaler.fit_transform(trnX)
    tstX = scaler.transform(tstX)

    lbl_enc = OneHotEncoder(categories=[np.sort(np.unique(y))], sparse=False)
    #lbl_enc = OneHotEncoder(sparse=False)
    lbl_enc.fit(trny.reshape(-1, 1))

    auto_var.set_intermidiate_variable("lbl_enc", lbl_enc)
    auto_var.set_intermidiate_variable("trnX", trnX)
    auto_var.set_intermidiate_variable("trny", trny)

    model = auto_var.get_var("model")
    auto_var.set_intermidiate_variable("model", model)
    model.fit(trnX, trny)

    if 'adv' in auto_var.get_variable_value("model") \
        or 'robustv1' in auto_var.get_variable_value("model"):
        auto_var.set_intermidiate_variable("trnX", model.augX)
        auto_var.set_intermidiate_variable("trny", model.augy)
        augX, augy = model.augX, model.augy
    elif 'robustv1nn' in auto_var.get_variable_value("model"):
        augX, augy = model.get_data()
        augy = (augy + 1) // 2
        auto_var.set_intermidiate_variable("trnX", augX)
        auto_var.set_intermidiate_variable("trny", augy)
        print(augy)
    else:
        augX, augy = None, None
    attack_model = auto_var.get_var("attack")

    #trn_perturbs = attack_model.perturb(trnX, y=trny, eps=eps)
    tst_perturbs = attack_model.perturb(tstX, y=tsty, eps=eps)

    ret = {}
    assert np.all(np.linalg.norm(tst_perturbs, axis=1, ord=ord) <= (eps + 1e-6)), (np.linalg.norm(tst_perturbs, axis=1, ord=ord), eps)
    temp_tstX = tstX + tst_perturbs
    tst_pred = model.predict(temp_tstX)

    ret['results'] = {
        'eps': eps,
        'tst_acc': (tst_pred == tsty).mean(),
    }
    print(ret[-1])

    if augX is not None:
        ret['aug_len'] = len(augX)
    ret['trnX_len'] = len(trnX)

    print(json.dumps(auto_var.var_value))
    print(json.dumps(ret))
    return ret

def main():
    auto_var.parse_argparse()
    auto_var.run_single_experiment(eps_accuracy)

if __name__ == '__main__':
    main()

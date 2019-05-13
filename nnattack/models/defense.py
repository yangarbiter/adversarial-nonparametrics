"""
"""
from __future__ import division

import numpy as np
from sklearn.neighbors import NearestNeighbors, KNeighborsClassifier
from scipy.spatial.distance import cdist

from .robust_nn.eps_separation import find_eps_separated_set
from ..variables import auto_var

def find_confident_label(X, Y, k, Delta):
    thres = 2*Delta
    neigh = NearestNeighbors(k)
    neigh.fit(X)
    nn = neigh.kneighbors(X, k, return_distance=False)
    Y_hat = np.array([[Y[j] for j in i] for i in nn])
    Y_hat = np.sum(Y_hat, axis=1)/k
    Y_hat = [0 if (abs(Y_hat[i]) < thres) else np.sign(Y_hat[i])
             for i in range(X.shape[0])]
    Y_hat = np.array(Y_hat)
    return Y_hat

def find_red_points(X, Y, Y_hat, eps, ord):
    n = X.shape[0]
    d = cdist(X, X, 'minkowski', ord)

    is_close = (d < eps)
    is_red = np.ones((n, 1))
    for i in range(n):
        for j in range(n):
            if (is_close[i, j]) and (Y_hat[i] != Y_hat[j]):
                is_red[i] = 0

            if Y_hat[i] != Y[i]:
                is_red[i] = 0
    red_pts = [np.array([X[i] for i in range(n) if is_red[i]]),
               np.array([Y[i] for i in range(n) if is_red[i]])]
    other_pts = [np.array([X[i] for i in range(n) if not is_red[i]]),
                 np.array([Y[i] for i in range(n) if not is_red[i]])]

    [X_red, Y_red] = [red_pts[0], red_pts[1]]
    [X_other, Y_other] = [other_pts[0], other_pts[1]]
    return X_red, Y_red, X_other, Y_other

def get_aug_v2(X, Y, Delta, delta, eps, ord):
    k = min(int(3*np.log(X.shape[0]/delta)/(np.log(2)*(Delta**2))), len(X))

    Y_hat = find_confident_label(X, Y, k, Delta)
    X_red, Y_red, X_other, Y_other = find_red_points(
            X, Y, eps=eps, Y_hat=Y_hat, ord=ord)

    print("X_red: ", X_red.shape)
    [X, Y] = [X_other, Y_other]
    print("X_other: ", X.shape)
    X, Y = find_eps_separated_set(X, eps/2, Y, ord=ord)

    if X_red.shape[0] > 0:
        X_train = np.concatenate([X, X_red])
        Y_train = np.concatenate([Y, Y_red])
    else:
        X_train = X
        Y_train = Y
    return X_train, Y_train

def get_aug_data(model, X, y, eps):
    """Augment the data for defense, returns the augmented data

    Arguments:
        model {Classifier} -- The original classifier model, model.train_type defines the way to augment the data.
            model.train_type == 'adv': adversarial training
            model.train_type == 'robustv2': adversarial pruning
            model.train_type == 'advPruning': Wang's defense for 1-NN
            model.train_type is None: Do nothing returns the original data
        X {ndarray, dim=2} -- feature vectors
        y {ndarray, dim=1} -- labels
        eps {float} -- defense strength

    Returns:
        augX {ndarray, dim=2} -- augmented feature vectors
        augy {ndarray, dim=1} -- augmented labels
    """
    if model.train_type in ['adv', 'advPruning', 'advPruningmin', 'robustv2']:
        if eps is None and model.eps is None:
            raise ValueError("eps should not be None with train type %s" % model.train_type)
        elif eps is None:
            eps = model.eps

    sep_measure = model.sep_measure if model.sep_measure else model.ord

    if model.train_type == 'adv':
        advX = model.attack_model.perturb(X, y=y, eps=eps)

        ind = np.where(np.linalg.norm(advX, axis=1) != 0)

        augX = np.vstack((X, X[ind]+advX[ind]))
        augy = np.concatenate((y, y[ind]))

    elif model.train_type == 'adv2':
        print(auto_var.var_value)
        model_name = auto_var.get_variable_name("model")
        aug_model_name = "_".join(model_name.split("_")[1:])
        print(aug_model_name)
        for _ in range(5):
            if "decision_tree" in model_name:
                auto_var.get_var_with_argument("model", "decision_tree_d5").fit(X, y=y)
            elif "rf" in model_name:
                auto_var.get_var_with_argument("model", "random_forest_100_d5").fit(X, y=y)
            elif "nn" in model_name:
                auto_var.get_var_with_argument("model", "_".join(model_name.split("_")[1:3])).fit(X, y=y)
            else:
                raise ValueError()
            advX = auto_var.get_var("attack").perturb(X, y=y, eps=eps)
            ind = np.where(np.linalg.norm(advX, axis=1) != 0)
            X = np.vstack((X, X[ind]+advX[ind]))
            y = np.concatenate((y, y[ind]))
            auto_var.set_intermidiate_variable("trnX", X)
            auto_var.set_intermidiate_variable("trny", y)
        augX, augy = X, y

    elif model.train_type == 'advPruningmin':
        if len(np.unique(y)) != 2:
            raise ValueError("Can only deal with number of classes = 2"
                             "got %d", len(np.unique(y)))
        y = y.astype(int)*2-1
        augX, augy = find_eps_separated_set(X, eps/2, y, 'min_measure')
        augy = (augy+1)//2

    elif model.train_type == 'advPruning':
        if len(np.unique(y)) != 2:
            raise ValueError("Can only deal with number of classes = 2"
                             "got %d", len(np.unique(y)))
        y = y.astype(int)*2-1
        augX, augy = find_eps_separated_set(X, eps/2, y, ord=sep_measure)
        augy = (augy+1)//2

    elif model.train_type == 'robustv2':
        if len(np.unique(y)) != 2:
            raise ValueError("Can only deal with number of classes = 2"
                             "got %d", len(np.unique(y)))
        y = y.astype(int)*2-1

        Delta = model.Delta
        delta = model.delta
        augX, augy = get_aug_v2(X, y, Delta, delta, eps, sep_measure)
        augy = (augy+1)//2

    elif model.train_type is None:
        augX, augy = X, y
    elif model.train_type == 'robust':
        augX, augy = X, y
    else:
        raise ValueError("Not supported training type %s", model.train_type)

    return augX, augy


"""
Not using
Download from https://raw.githubusercontent.com/EricYizhenWang/robust_nn_icml/master/robust_1nn.py
"""

from __future__ import division
import numpy as np
from sklearn.neighbors import NearestNeighbors, KNeighborsClassifier
from scipy.spatial.distance import cdist

from .eps_separation import find_eps_separated_set

def find_confident_label(X, Y, k, Delta):
    thres = 2*Delta
    #print(thres, k)
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

class Robust_1NN():
    def __init__(self, Delta, delta, ord, train_type='robust_v2'):
        #self.X = X
        #self.Y = Y * 2 -1 # to 1 -1
        self.ord = ord
        self.Delta = Delta
        self.delta = delta

        self.train_type = train_type

    def find_confident_label(self):
        [X, Y, k, delta, n] = [self.X, self.Y,
                               self.k, self.delta,
                               self.n]
        thres = 2*self.Delta
        #print(thres, k)
        neigh = NearestNeighbors(k)
        neigh.fit(X)
        nn = neigh.kneighbors(X, k, return_distance=False)
        Y_hat = np.array([[Y[j] for j in i] for i in nn])
        Y_hat = np.sum(Y_hat, axis=1)/k
        Y_hat = [0 if (abs(Y_hat[i])<thres) else np.sign(Y_hat[i])
                 for i in range(n)]
        Y_hat = np.array(Y_hat)
        self.Y_hat = Y_hat
        return Y_hat

    def find_red_points(self, eps):
        [X, Y, d, r, F] = [self.X, self.Y, self.d, eps, self.Y_hat]
        n = X.shape[0]
        # TODO this fits only for L2 norm
        is_close = (d<r)
        is_red = np.ones((n, 1))
        for i in range(n):
            for j in range(n):
                if (is_close[i, j]) and (F[i]!=F[j]):
                    is_red[i]=0
                if (F[i]!=Y[i]):
                    is_red[i]=0
        red_pts = [np.array([X[i] for i in range(n) if is_red[i]]),
                  np.array([Y[i] for i in range(n) if is_red[i]])]
        other_pts = [np.array([X[i] for i in range(n) if not is_red[i]]),
                  np.array([Y[i] for i in range(n) if not is_red[i]])]
        [self.X_red, self.Y_red] = [red_pts[0], red_pts[1]]
        [self.X_other, self.Y_other] = [other_pts[0], other_pts[1]]
        return [red_pts, other_pts]

    def find_robust_training_set(self, eps):
        if self.train_type == 'robust_v2':
            self.find_confident_label()
            self.find_red_points(eps)
            [X, Y, r] = [self.X_other, self.Y_other, eps]
        else:
            [X, Y, r] = [self.X, self.Y, eps]
        X, Y = find_eps_separated_set(X, eps/2, Y, ord=self.ord)
        if self.X_red.shape[0] > 0:
            self.X_train = np.concatenate([X, self.X_red])
            self.Y_train = np.concatenate([Y, self.Y_red])
        else:
            self.X_train = X
            self.Y_train = Y
        return self.X_train, self.Y_train

    def fit(self, X, Y, eps:float):
        if eps == 0:
            self.neigh = KNeighborsClassifier(n_neighbors=1)
            self.neigh.fit(X, Y)
            self.augX, self.augy = X, Y
        else:
            self.X = X
            self.Y = Y * 2 -1 # to 1 -1
            self.n = X.shape[0]
            #self.d = np.array([[np.linalg.norm(a-b, ord=self.ord) for a in X] for b in X])
            self.d = cdist(X, X, 'minkowski', self.ord)
            self.k = min(int(3*np.log(self.n/self.delta)/(np.log(2)*(self.Delta**2))), self.n)

            self.find_robust_training_set(eps)
            [X, Y] = [self.X_train, (self.Y_train + 1)//2]
            self.neigh = KNeighborsClassifier(n_neighbors=1)
            self.neigh.fit(X, Y)
            self.augX, self.augy = X, Y

    def predict(self, X):
        return self.neigh.predict(X)

    def get_clf(self):
        return self.neigh

    def get_data(self):
        return [self.X_train, (self.Y_train + 1) // 2]


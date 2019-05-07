"""
Not using
Download from https://raw.githubusercontent.com/EricYizhenWang/robust_nn_icml/master/robust_1nn.py
"""

from __future__ import division
import numpy as np
from sklearn.neighbors import NearestNeighbors, KNeighborsClassifier

from .eps_separation import find_eps_separated_set

class RobustNN:
    def __init__(self, Delta, delta, r):
        self.r = r
        self.Delta = Delta
        self.delta = delta

    def _setup(self, X, Y):
        self.X = X
        self.Y = Y
        n = X.shape[0]
        self.n = n
        self.k = min(int(3*np.log(n/self.delta)/(np.log(2)*(self.Delta**2))), n)
        d = np.array([[np.linalg.norm(a-b) for a in X] for b in X])
        self.d = d

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

    def find_red_points(self):
        [X, Y, d, r, F] = [self.X, self.Y, self.d, self.r, self.Y_hat]
        n = X.shape[0]
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

    def find_robust_training_set(self):
        self.find_confident_label()
        self.find_red_points()
        [X, Y, r] = [self.X_other, self.Y_other, self.r]
        X, Y = find_eps_separated_set(X, r/2, Y)
        if self.X_red.shape[0] > 0:
            self.X_train = np.concatenate([X, self.X_red])
            self.Y_train = np.concatenate([Y, self.Y_red])
        else:
            self.X_train = X
            self.Y_train = Y
        return self.X_train, self.Y_train

    def fit(self, X, Y):
        self._setup(X, Y)
        self.find_robust_training_set()
        [X, Y] = [self.X_train, self.Y_train]
        self.neigh = KNeighborsClassifier(n_neighbors=1)
        self.neigh.fit(X, Y)

    def predict(self, X):
        return self.neigh.predict(X)

    def get_clf(self):
        return self.neigh

    def get_data(self):
        return [self.X_train, self.Y_train]


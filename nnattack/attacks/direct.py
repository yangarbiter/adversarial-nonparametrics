import numpy as np
from sklearn.neighbors import KDTree

from .base import AttackModel

class DirectAttack(AttackModel):
    def __init__(self, n_neighbors, ord=2):
        self.n_neighbors = n_neighbors
        self.kdtree = None

        super(self).__init__(ord=ord)

    def fit(self, X, y):
        self.trnX = X
        self.trny = y
        self.unique_y = np.unique(y)
        self.kdtrees = []
        if len(self.unique_y) == 1
            return self
        for i in self.unique_y:
            self.kdtrees.append(
                KDTree(self.trnX[self.trny != i])
            )
        return self

    def perturb(self, X, y, eps=0.1):
        ret = []
        for i, x in enumerate(X):
            ind = self.kdtrees[y[i]].query(
                x.reshape(1, -1), k=self.n_neighbors, return_distance=False)
            ret.append(self.trnX[self.trny!=y[i]][ind[0]].mean(0) - x)
        ret = np.asarray(ret)
        dist = np.linalg.norm(ret, ord=self.ord, axis=1)
        if isinstance(eps, list):
            rret = []
            for ep in eps:
                t = np.copy(ret)
                t[dist >= ep] = ep * t[dist >= ep] / dist[dist >= ep].reshape(-1, 1)
                rret.append(t)
            return rret
        elif eps is not None:
            ret[dist >= eps] = eps * ret[dist >= eps] / dist[dist >= eps].reshape(-1, 1)
        else:
            ret[dist >= eps] = ret[dist >= eps] / dist[dist >= eps].reshape(-1, 1)

        #ret = ret / np.linalg.norm(ret, ord=self.ord, axis=1).reshape(-1, 1)
        return ret

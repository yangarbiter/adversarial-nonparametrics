import numpy as np
from sklearn.svm import LinearSVC
from cvxopt import matrix, solvers
import cvxopt.glpk
import cvxopt

from .robust_nn.eps_separation import find_eps_separated_set
from .sklr import get_sol_l2, get_sol_linf

class SkLinSVC(LinearSVC):
    def __init__(self, **kwargs):
        print(kwargs)
        self.ord = kwargs.pop("ord", np.inf)
        self.eps = kwargs.pop("eps", 0.1)
        self.train_type = kwargs.pop("train_type", None)
        super().__init__(**kwargs)

    def fit(self, X, y):
        print("original X", np.shape(X), len(y))
        if self.train_type == 'adv':
            clf = LinearSVC().fit(X, y)
            advX = self._get_perturb(X, y=y, eps=self.eps, model=clf)

            ind = np.where(np.linalg.norm(advX, axis=1) != 0)

            self.augX = np.vstack((X, X[ind]+advX[ind]))
            self.augy = np.concatenate((y, y[ind]))
            print("number of augX", np.shape(self.augX), len(self.augy))
        elif self.train_type == 'advPruning':
            y = y.astype(int)*2-1
            self.augX, self.augy = find_eps_separated_set(X, self.eps/2, y)
            self.augy = (self.augy+1)//2
            print("number of augX", np.shape(self.augX), len(self.augy))
        elif self.train_type is None:
            return super().fit(X, y)
        else:
            raise ValueError("Not supported training type %s", self.train_type)


        return super().fit(self.augX, self.augy)

    def _get_perturb(self, X, y, eps, model):
        pert_X = np.zeros_like(X)

        if self.ord == 2:
            get_sol_fn = get_sol_l2
        elif self.ord == np.inf:
            get_sol_fn = get_sol_linf
        else:
            raise ValueError("ord %s not supported", self.ord)

        G = model.coef_.reshape(1, -1)
        h = model.intercept_
        for i, (x, yi) in enumerate(zip(X, y)):
            if model.predict([x]) != yi:
                continue
            if model.decision_function(x.reshape(1, -1))[0] > 0:
                # Gx + h
                pert_x = get_sol_fn(x, G, -h)
            else:
                pert_x = get_sol_fn(x, -G, h)

            assert model.predict([x + pert_x])[0] != yi
            pert_X[i, :] = pert_x

        if isinstance(eps, list):
            rret = []
            norms = np.linalg.norm(pert_X, axis=1, ord=self.ord)
            for ep in eps:
                t = np.copy(pert_X)
                t[norms > ep, :] = 0
                rret.append(t)
            return rret
        elif eps is not None:
            pert_X[np.linalg.norm(pert_X, axis=1, ord=self.ord) > eps, :] = 0
            return pert_X
        else:
            return pert_X

    def perturb(self, X:np.array, y:np.array=None, eps=0.1):
        return self._get_perturb(X, y, eps, self)

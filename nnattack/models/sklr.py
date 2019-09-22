import numpy as np
from sklearn.linear_model import LogisticRegression
from joblib import Parallel, delayed
from cvxopt import matrix, solvers
import cvxopt.glpk
import cvxopt

from .robust_nn.eps_separation import find_eps_separated_set

solvers.options['show_progress'] = False
cvxopt.glpk.options["msg_lev"] = "GLP_MSG_OFF"

class SkLr(LogisticRegression):
    def __init__(self, **kwargs):
        print(kwargs)
        self.ord = kwargs.pop("ord", np.inf)
        self.eps = kwargs.pop("eps", 0.1)
        self.train_type = kwargs.pop("train_type", None)
        super().__init__(**kwargs)

    def fit(self, X, y):
        print("original X", np.shape(X), len(y))
        if self.train_type == 'adv':
            clf = LogisticRegression().fit(X, y)
            advX = self._get_perturb(X, y=y, eps=self.eps, model=clf)

            ind = np.where(np.linalg.norm(advX, axis=1) != 0)

            self.augX = np.vstack((X, X[ind]+advX[ind]))
            self.augy = np.concatenate((y, y[ind]))
            print("number of augX", np.shape(self.augX), len(self.augy))
        elif self.train_type == 'adv2':
            for i in range(10):
                clf = LogisticRegression().fit(X, y)
                advX = self._get_perturb(X, y=y, eps=self.eps, model=clf)

                ind = np.where(np.linalg.norm(advX, axis=1) != 0)

                X = np.vstack((X, X[ind]+advX[ind]))
                y = np.concatenate((y, y[ind]))
            self.augX, self.augy = X, y
            print("number of augX", np.shape(self.augX), len(self.augy))
        elif self.train_type == 'advPruning':
            y = y.astype(int)*2-1
            self.augX, self.augy = find_eps_separated_set(X, self.eps/2, y,
                                                          ord=self.ord)
            self.augy = (self.augy+1)//2
            print("number of augX", np.shape(self.augX), len(self.augy))
        elif self.train_type is None:
            return super().fit(X, y)
        else:
            raise ValueError("Not supported training type %s", self.train_type)


        return super().fit(self.augX, self.augy)

    def _get_perturb(self, X, y, eps, model):
        assert model.coef_.shape[0] == 1

        pert_X = np.zeros_like(X)

        if self.ord == 2:
            get_sol_fn = get_sol_l2
        elif self.ord == np.inf:
            get_sol_fn = get_sol_linf
        else:
            raise ValueError("ord %s not supported", self.ord)

        G = model.coef_
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



def get_sol_l2(target_x, G, h):
    G, h = matrix(G, tc='d'), matrix(h, tc='d')
    fet_dim = target_x.shape[0]
    temph = h - 1e-4

    Q = 2 * matrix(np.eye(fet_dim), tc='d')

    q = matrix(-2*target_x, tc='d')

    sol = solvers.qp(P=Q, q=q, G=G, h=temph)

    if sol['status'] == 'optimal':
        ret = np.array(sol['x'], np.float64).reshape(-1)
        return ret - target_x
    else:
        raise ValueError

def get_sol_linf(target_x, G, h):
    fet_dim = target_x.shape[0]
    c = matrix(np.concatenate((np.zeros(fet_dim), np.ones(1))), tc='d')

    G2 = np.hstack((np.eye(fet_dim), -np.ones((fet_dim, 1))))
    G3 = np.hstack((-np.eye(fet_dim), -np.ones((fet_dim, 1))))
    G = np.hstack((G, np.zeros((G.shape[0], 1))))
    G = np.vstack((G, G2, G3))
    h = np.concatenate((h, target_x, -target_x))

    G, h = matrix(G, tc='d'), matrix(h, tc='d')
    temph = h - 1e-4

    lp_sol = solvers.lp(c=c, G=G, h=temph, solver='glpk')
    if lp_sol['status'] == 'optimal':
        sol = np.array(lp_sol['x']).reshape(-1)[:-1]
        return sol - target_x
    else:
        raise ValueError

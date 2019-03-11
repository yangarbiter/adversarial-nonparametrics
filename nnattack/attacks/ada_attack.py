from copy import deepcopy
import itertools

import numpy as np
from sklearn.ensemble import AdaBoostClassifier
from joblib import Parallel, delayed
from cvxopt import matrix, solvers
import cvxopt.glpk
import cvxopt

solvers.options['maxiters'] = 30
solvers.options['show_progress'] = False
solvers.options['refinement'] = 0
solvers.options['feastol'] = 1e-7
solvers.options['abstol'] = 1e-7
solvers.options['reltol'] = 1e-7
cvxopt.glpk.options["msg_lev"] = "GLP_MSG_OFF"


def get_sol_l2(target_x, target_y, constraints, perm_values, tar_estimators, transformer):
    fet_dim = target_x.shape[1]
    temp = (target_x, np.inf)

    for i, v in enumerate(perm_values):
        if np.sign(v) == (target_y * 2 - 1):
            continue

        G, h = constraints[i]
        G, h = matrix(G, tc='d'), matrix(h, tc='d')
        temph = h - 1e-4

        c = matrix(np.concatenate((np.zeros(fet_dim), np.ones(1))), tc='d')

        Q = 2 * matrix(np.eye(fet_dim), tc='d')
        T = matrix(transformer.astype(np.float64), tc='d')

        G = G * T
        q = matrix(-2*target_x, tc='d')

        sol = solvers.qp(P=Q, q=q, G=G, h=temph)

        if sol['status'] == 'optimal':
            ret = np.array(sol['x'], np.float64).reshape(-1)
            eps = np.linalg.norm(ret - target_x, ord=2)
            if eps < temp[1]:
                temp = (ret, eps)

    if temp [1] != np.inf:
        return temp[0] - target_x
    else:
        return np.zeros_like(target_x)

def get_sol_linf(target_x, target_y, constraints, perm_values, tar_estimators, transformer):
    fet_dim = target_x.shape[1]
    temp = (target_x, np.inf)

    for i, v in enumerate(perm_values):
        if np.sign(v) == (target_y * 2 - 1):
            continue
        G, h = constraints[i]

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
            eps = np.linalg.norm(sol - target_x, ord=np.inf)
            if eps < temp[1]:
                temp = (sol, eps)

    if temp [1] != np.inf:
        return temp[0] - target_x
    else:
        return np.zeros_like(target_x)


class ADAAttack():
    def __init__(self, clf: AdaBoostClassifier, n_features: int, ord, transformer,
                 random_state):
        self.clf = clf
        self.ord = ord
        self.transformer = transformer
        self.random_state = random_state
        constraints = []

        #estimator_weight = clf.learning_rate * (
        #    np.log((1. - clf.estimator_error) / estimator_error))

        for tree in clf.estimators_:
            assert len(tree.tree_.feature) == 3 # max_depth = 1
            sol = (np.argmax(tree.tree_.value[1]), np.argmax(tree.tree_.value[2]))
            assert ((sol == (0, 1)) or (sol == (1, 0)) or (sol == (0, 0)) or (sol == (1, 1)))
            constraints.append((tree.tree_.feature[0], tree.tree_.threshold[0], sol))
        self.tar_estimators = sorted(zip(clf.estimator_weights_, constraints),
                                     key=lambda x: x[0], reverse=True)
        print(self.tar_estimators)

        not_possible_constraint_ids = []
        self.constraints: list = []
        bin_permutations: list = list(itertools.product([-1, 1], repeat=len(self.tar_estimators)))
        self.perm_values = [np.dot(clf.estimator_weights_, np.array(i)) for i in bin_permutations]
        for idx, perm in enumerate(bin_permutations):
            G, h = [], []
            for i, p in enumerate(perm):
                G.append(np.zeros(n_features))
                sol = self.tar_estimators[i][1][2]
                if (sol == (0, 0) and p == 0) or (sol == (1, 1) and p == 1):
                    continue
                elif (sol == (0, 0) and p == 1) or (sol == (1, 1) and p == 0):
                    G, h = None, None
                    not_possible_constraint_ids.append(idx)
                elif (p <= 0 and sol[0] == 0) or (p > 1 and sol[0] == 1):
                    G[-1][self.tar_estimators[i][1][0]] = 1
                    h.append(self.tar_estimators[i][1][1])
                else:
                    G[-1][self.tar_estimators[i][1][0]] = -1
                    h.append(-self.tar_estimators[i][1][1])
            if G is not None:
                self.constraints.append((np.asarray(G), np.asarray(h)))
        
        for i in not_possible_constraint_ids[::-1]:
            del self.perm_values[i]

        assert len(self.perm_values) == len(self.constraints)

    def fit(self, X, y):
        pass

    def perturb(self, X, y, eps=0.1):
        pert_X = np.zeros_like(X)
        import ipdb; ipdb.set_trace()
        pred_y = self.clf.predict(X)


        if self.ord == 2:
            get_sol_fn = get_sol_l2
        elif self.ord == np.inf:
            get_sol_fn = get_sol_linf
        else:
            raise ValueError("ord %s not supported", self.ord)

        for sample_id in range(len(X)):
            if pred_y[sample_id] != y[sample_id]:
               continue
            target_x = X[sample_id]
            pert_x = get_sol_fn(target_x, y[sample_id],
                                self.constraints,
                                self.perm_values,
                                self.tar_estimators,
                                self.transformer)
            if np.linalg.norm(pert_x) != 0:
                assert self.clf.predict([X[sample_id] + pert_x])[0] != y[sample_id]
                pert_X[sample_id, :] = pert_x # pylint: disable=unsupported-assignment-operation
        
        if isinstance(eps, list):
            rret = []
            norms = np.linalg.norm(pert_X, axis=1, ord=self.ord)
            for ep in eps:
                t = np.copy(pert_X)
                t[norms > ep, :] = 0
                rret.append(t)
            return rret
        elif eps is not None:
            pert_X[np.linalg.norm(pert_X, axis=1, ord=self.ord) > eps, :] = 0 # pylint: disable=unsupported-assignment-operation
            return pert_X
        else:
            return pert_X


        

from itertools import product, permutations

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KDTree
from bistiming import IterTimer
import joblib
from joblib import Parallel, delayed

from .base import AttackModel
from .dt_opt import get_tree_constraints
from .nn_attack import solve_lp, solve_qp

def union_constraints(G, h):
    assert np.all(np.abs(G).sum(1) == np.ones(len(G)))
    #import ipdb; ipdb.set_trace()
    if len(np.shape(G)) <= 1:
        return np.array([]), np.array([])
    n_dim = np.shape(G)[1]

    r = [None for i in range(n_dim*2)]
    for Gi, hi in zip(G, h):
        if Gi.sum() == 1:
            idx = np.where(Gi == 1)[0][0]
            r[idx] = hi if r[idx] is None else min(r[idx], hi)
        elif Gi.sum() == -1:
            idx = np.where(Gi == -1)[0][0] + n_dim
            r[idx] = hi if r[idx] is None else min(r[idx], hi)
        else:
            raise ValueError()

    rG, rh = [], []
    for i in range(len(r)):
        if r[i] is not None:
            temp = np.zeros(n_dim)
            if i < n_dim:
                temp[i] = 1
            else:
                temp[i-n_dim] = -1
            rG.append(temp)
            rh.append(r[i])

    return np.array(rG), np.array(rh)

def tree_instance_constraint(tree_clf, X):
    node_indicator = tree_clf.decision_path(X)
    leave_id = tree_clf.apply(X)
    feature = tree_clf.tree_.feature
    threshold = tree_clf.tree_.threshold
    n_dims = X.shape[1]

    Gs, hs = [], []
    for sample_id in range(len(X)):
        node_index = node_indicator.indices[node_indicator.indptr[sample_id]:
                                            node_indicator.indptr[sample_id + 1]]
        G, h = [], []
        for node_id in node_index:
            if leave_id[sample_id] == node_id:
                break

            G.append(np.zeros(n_dims))
            if (X[sample_id, feature[node_id]] <= threshold[node_id]):
                #threshold_sign = "<="
                G[-1][feature[node_id]] = 1
                h.append(threshold[node_id])
            else:
                #threshold_sign = ">"
                G[-1][feature[node_id]] = -1
                h.append(-threshold[node_id] - 1e-9) # excluding equality

        Gs.append(np.array(G))
        hs.append(np.array(h))

    return Gs, hs

def rev_get_sol_linf(target_x, target_y: int, pred_trn_y, regions, clf,
        trnX=None):
    fet_dim = np.shape(target_x)[0]
    candidates = []
    for i, (G, h) in enumerate(regions):
        c = np.concatenate((np.zeros(fet_dim), np.ones(1))).reshape((-1, 1))

        G2 = np.hstack((np.eye(fet_dim), -np.ones((fet_dim, 1))))
        G3 = np.hstack((-np.eye(fet_dim), -np.ones((fet_dim, 1))))
        G = np.hstack((G, np.zeros((G.shape[0], 1))))
        G = np.vstack((G, G2, G3))
        h = np.concatenate((h, target_x, -target_x))

        temph = (h - 1e-6).reshape((-1, 1))

        if trnX is None:
            status, sol = solve_lp(c, G, temph, len(c))
        else:
            init_x = np.concatenate((
                trnX[i],
                [np.linalg.norm(trnX[i]-target_x, ord=np.inf)])).reshape((-1, 1))
            status, sol = solve_lp(c, G, temph, len(c), init_x=init_x)

        if status == 'optimal':
            ret = np.array(sol).reshape(-1)[:-1]
            #print(clf.predict([ret])[0], target_y)
            #print(np.dot(ori_G, target_x) <= ori_h)
            #print(np.dot(ori_G, ret) <= ori_h)
            #print(np.dot(ori_G, trnX[np.where(pred_trn_y != target_y)[0][i]]) <= ori_h)
            #print([est.predict([ret]) for est in clf.estimators_])
            #print([est.predict([trnX[np.where(pred_trn_y != target_y)[0][i]]]) for est in clf.estimators_])
            assert clf.predict([ret])[0] != target_y
            candidates.append(ret - target_x)
        elif status == 'infeasible_inaccurate':
            #print(status)
            pass
            #import ipdb; ipdb.set_trace()
            #ret = np.array(sol).reshape(-1)[:-1]
            #if clf.predict([ret])[0] != target_y:
            #    candidates.append(ret)
        else:
            print(status)
            #raise ValueError()

    norms = np.linalg.norm(candidates, ord=np.inf, axis=1)
    return candidates[norms.argmin()]



class RFAttack(AttackModel):
    def __init__(self, trnX, trny, clf: RandomForestClassifier, ord,
            method: str, n_searches=-1, random_state=None):
        super().__init__(ord=ord)
        paths, constraints = [], []
        self.clf = clf
        self.method = method
        self.n_searches = n_searches
        self.trnX = trnX
        self.trny = trny
        self.random_state = random_state
        if self.n_searches != -1:
            self.kd_tree = KDTree(self.trnX)
        else:
            self.kd_tree = None

        if self.method == 'all':
            for tree_clf in clf.estimators_:
                path, constraint = get_tree_constraints(tree_clf)
                paths.append(path)
                constraints.append(constraint)

            n_classes = clf.n_classes_
            n_estimators = len(clf.estimators_)
            self.regions = []
            self.region_preds = []

            for res in product(range(n_classes), repeat=n_estimators):
                perm_consts = [list() for _ in range(n_estimators)]

                for i in range(n_estimators):
                    value = clf.estimators_[i].tree_.value
                    path = paths[i]
                    constraint = constraints[i]

                    for p in range(len(path)):
                        if np.argmax(value[path[p][-1]]) == res[i]:
                            perm_consts[i].append(constraint[p])

                for pro in product(*perm_consts):
                    self.region_preds.append(np.argmax(np.bincount(res)))
                    self.regions.append(
                        union_constraints(
                            np.vstack([j[0] for j in pro]),
                            np.concatenate([j[1] for j in pro]),
                        )
                    )

            print(f"number of regions: {len(self.regions)}")

        elif self.method == 'rev':
            Gss, hss = [list() for _ in trnX], [list() for _ in trnX]
            for tree_clf in clf.estimators_:
                Gs, hs = tree_instance_constraint(tree_clf, trnX)
                for i, (G, h) in enumerate(zip(Gs, hs)):
                    Gss[i].append(G)
                    hss[i].append(h)
            self.regions = [union_constraints(np.vstack(Gs), np.concatenate(hs)) \
                            for Gs, hs in zip(Gss, hss)]

            for i in range(len(trnX)):
                G, h = self.regions[i]
                assert np.all(np.dot(G, trnX[i]) <= h)
        else:
            raise ValueError("Not supported method: %s", self.method)

    def perturb(self, X, y, eps=0.1):
        pert_X = np.zeros_like(X)

        if self.ord == 2:
            get_sol_fn = rev_get_sol_l2
        elif self.ord == np.inf:
            get_sol_fn = rev_get_sol_linf
        else:
            raise ValueError("ord %s not supported", self.ord)

        pred_y = self.clf.predict(X)
        pred_trn_y = self.clf.predict(self.trnX)

        with IterTimer("Perturbing", len(X)) as timer:
            for sample_id in range(len(X)):
                timer.update(sample_id)
                if pred_y[sample_id] != y[sample_id]:
                    continue
                target_x, target_y = X[sample_id], y[sample_id]

                if self.method == 'all':
                    temp_regions = [self.regions[i] \
                            for i in range(len(self.regions)) \
                                if self.region_preds[i] != target_y]
                    pert_x = get_sol_fn(target_x, y[sample_id],
                            pred_trn_y, temp_regions, self.clf)
                elif self.method == 'rev':
                    if self.n_searches != -1:
                        ind = self.kd_tree.query(
                                target_x.reshape((1, -1)),
                                k=len(self.trnX),
                                return_distance=False)[0]
                        ind = list(filter(lambda x: pred_trn_y[x] != target_y, ind))[:self.n_searches]
                    else:
                        ind = list(filter(lambda x: pred_trn_y[x] != target_y, np.arange(len(self.trnX))))
                    temp_regions = [self.regions[i] for i in ind]
                    pert_x = get_sol_fn(target_x, y[sample_id],
                            pred_trn_y, temp_regions,
                            self.clf, self.trnX[ind])
                else:
                    raise ValueError("Not supported method %s", self.method)

                pert_X[sample_id] = pert_x

                if np.linalg.norm(pert_x) != 0:
                    assert self.clf.predict([X[sample_id] + pert_x])[0] != y[sample_id]
                    pert_X[sample_id, :] = pert_x
                else:
                    raise ValueError("shouldn't happen")
        
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

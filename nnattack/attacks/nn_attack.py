import itertools
import tempfile
import os
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from cvxopt import matrix, solvers
import cvxopt.glpk
#import cvxopt
import numpy as np
from sklearn.neighbors import KDTree
import scipy
import joblib
from joblib import Parallel, delayed
from sklearn.neighbors import KNeighborsClassifier
from bistiming import IterTimer

from .cutils import c_get_half_space, get_all_half_spaces, get_constraints, check_feasibility

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

#import cvxopt.msk
#from mosek import iparam
#solvers.options['MOSEK'] = {iparam.log: 0}

#import mosek
#msk.options = {mosek.iparam.log: 0}
#solvers.options['solver'] = 'glpk'
#solvers.options['maxiters'] = 30
solvers.options['show_progress'] = False
#solvers.options['refinement'] = 0
#solvers.options['feastol'] = 1e-7
#solvers.options['abstol'] = 1e-7
#solvers.options['reltol'] = 1e-7
cvxopt.glpk.options["msg_lev"] = "GLP_MSG_OFF"

CONSTRAINTTOL = 5e-6

def get_half_space(a, b):
    w = (b - a)
    c = np.dot(w.T, (a + b) / 2)
    sign = -np.sign(np.dot(w.T, b) - c)
    w = sign * w
    c = sign * c
    return [w, c]

glob_trnX = None
glob_trny = None

DEBUG = False

import cvxpy as cp

def solve_lp(c, G, h, n, init_x=None, n_jobs=1):
    #c = np.array(c)
    #G, h = np.array(G), np.array(h)
    options = {'threads': n_jobs}
    x = cp.Variable(shape=(n, 1))
    obj = cp.Minimize(c.T * x)
    constraints = [G*x <= h]
    prob = cp.Problem(obj, constraints)
    if init_x is not None:
        x.value = init_x
        prob.solve(solver=cp.GUROBI, warm_start=True)
    else:
        prob.solve(solver=cp.GUROBI)
    return prob.status, x.value

def solve_qp(Q, q, G, h, n):
    x = cp.Variable(shape=(n, 1))
    obj = cp.Minimize(cp.sum(cp.square(x)) + q.T * x)
    constraints = [G*x <= h]
    prob = cp.Problem(obj, constraints)
    try:
        prob.solve(solver=cp.GUROBI)
    except cp.error.SolverError:
        try:
            prob.solve(solver=cp.CVXOPT)
        except cp.error.SolverError:
            logger.error("Rare")
            return False, x.value
    return prob.status, x.value


#@profile
def get_sol(target_x, tuple_x, faropp, kdtree, transformer,
        glob_trnX, glob_trny, init_x=None):
    tuple_x = np.asarray(tuple_x)
    trnX = glob_trnX.dot(transformer.T)
    emb_tar = target_x.dot(transformer.T)
    G, h, _ = get_constraints(trnX, tuple_x, kdtree, faropp, emb_tar)
    G, h = matrix(G, tc='d'), matrix(h, tc='d')

    #assert (transformer.shape[1] == glob_trnX.shape[1])
    #n_emb = transformer.shape[0]
    n_fets = target_x.shape[0]

    #Q = 2 * matrix(np.eye(n_emb), tc='d')
    Q = 2 * matrix(np.eye(n_fets), tc='d')
    #Q = 2 * matrix(np.dot(np.dot(transformer.T, np.eye(n_emb)), transformer), tc='d')
    T = matrix(transformer.astype(np.float64), tc='d')

    G = G * T
    #Q = T.trans() * Q * T
    #q = matrix(-2*target_x.dot(transformer.T).dot(transformer), tc='d')
    q = matrix(-2*target_x, tc='d')

    temph = h - CONSTRAINTTOL # make sure all constraints are met

    status, sol = solve_qp(np.array(Q), np.array(q), np.array(G),
                           np.array(temph), n_fets)
    if status == 'optimal':
        ret = sol.reshape(-1)
        return True, ret
    else:
        return False, None

    #try:
    #    c = matrix(np.zeros(target_x.shape[0]), tc='d')
    #    if init_x is None:
    #        lp_sol = solvers.lp(c=c, G=G, h=temph, solver='glpk')
    #        if lp_sol['status'] == 'optimal':
    #            init_x = lp_sol['x']
    #        else:
    #            init_x = None
    #    if init_x is not None:
    #        #sol2 = solve_qp(np.array(Q), np.array(q).flatten(), np.array(G).T,
    #        #        np.array(temph).flatten())
    #        sol = solvers.qp(P=Q, q=q, G=G, h=temph, initvals=init_x)
    #        if sol['status'] == 'optimal':
    #            ret = np.array(sol['x'], np.float64).reshape(-1)
    #            if DEBUG:
    #                # sanity check for the correctness of objective
    #                print('1', sol['primal objective'] + np.dot(target_x, target_x))
    #                print('2', np.linalg.norm(target_x - ret, ord=2)**2)
    #                #print(sol['primal objective'], np.dot(target_x.dot(transformer.T), target_x.dot(transformer.T)))
    #                #print(sol['primal objective'] + np.dot(target_x.dot(transformer.T), target_x.dot(transformer.T)))
    #                #print(np.linalg.norm(target_x.dot(transformer.T) - ret.dot(transformer.T))**2)
    #                #assert np.isclose(np.linalg.norm(target_x.dot(transformer.T) - ret.dot(transformer.T))**2,
    #                #        sol['primal objective'] + np.dot(target_x.dot(transformer.T), target_x.dot(transformer.T)))
    #                assert np.isclose(np.linalg.norm(target_x - ret, ord=2)**2, sol['primal objective'] + np.dot(target_x, target_x))
    #                # check if constraints are all met
    #                h = np.array(h).flatten()
    #                G = np.array(G)
    #                a = check_feasibility(G, h, ret, G.shape[0], G.shape[1])
    #                assert a
    #            return True, ret
    #        return False, np.array(init_x, np.float64).reshape(-1)
    #    else:
    #        return False, None
    #except ValueError:
    #    #logger.warning("solver error")
    #    return False, None

def sol_sat_constraints(G, h):
    fet_dim = G.shape[1]
    c = matrix(np.zeros(fet_dim), tc='d')
    G = matrix(G, tc='d')
    temph = matrix(h - CONSTRAINTTOL, tc='d')
    sol = solvers.lp(c=c, G=G, h=temph, solver='glpk')
    return (sol['status'] == 'optimal')


def get_sol_l1(target_x, tuple_x, faropp, kdtree, transformer, glob_trnX,
        glob_trny, init_x=None):
    tuple_x = np.asarray(tuple_x)
    fet_dim = target_x.shape[0]
    #n_emb = transformer.shape[0]

    emb_tar = target_x.dot(transformer.T)
    trnX = glob_trnX.dot(transformer.T)
    G, h, dist = get_constraints(trnX, tuple_x, kdtree, faropp, emb_tar)
    G = np.dot(G, transformer)

    if init_x is None and not sol_sat_constraints(G, h):
        return False, None

    c = matrix(np.concatenate((np.zeros(fet_dim), np.ones(fet_dim))), tc='d')

    G = np.hstack((G, np.zeros((G.shape[0], fet_dim))))
    #G = np.vstack((G, np.hstack((transformer, -np.eye(fet_dim)))))
    #G = np.vstack((G, np.hstack((-transformer, -np.eye(fet_dim)))))
    G = np.vstack((G, np.hstack((np.eye(fet_dim), -np.eye(fet_dim)))))
    G = np.vstack((G, np.hstack((-np.eye(fet_dim), -np.eye(fet_dim)))))

    #h = np.concatenate((h, emb_tar, -emb_tar))
    h = np.concatenate((h, target_x, -target_x))

    G, h = matrix(G, tc='d'), matrix(h, tc='d')

    temph = h - CONSTRAINTTOL
    if init_x is not None:
        sol = solvers.lp(c=c, G=G, h=temph, solver='glpk',
                         initvals=init_x)
    else:
        sol = solvers.lp(c=c, G=G, h=temph, solver='glpk')

    if sol['status'] == 'optimal':
        ret = np.array(sol['x']).reshape(-1)
        ### sanity check for the correctness of objective
        if DEBUG:
            # check if constraints are all met
            h = np.array(h).flatten()
            G = np.array(G)
            a = check_feasibility(G, h, ret, G.shape[0], G.shape[1])
            print(a)
            print('1', sol['primal objective'])
            print('2', np.linalg.norm(target_x - ret[:len(ret)//2], ord=1))
            print('3', np.linalg.norm(target_x.dot(transformer.T) - (ret[:len(ret)//2]).dot(transformer.T), ord=1))
            #print(target_x.dot(transformer.T))
            #print(ret)
            #assert np.isclose(np.linalg.norm(target_x.dot(transformer.T) - (ret[:len(ret)//2]).dot(transformer.T), ord=1),
            #                  sol['primal objective'], rtol=CONSTRAINTTOL)
            assert np.isclose(np.linalg.norm(target_x - ret[:len(ret)//2], ord=1),
                              sol['primal objective'], rtol=CONSTRAINTTOL)
        return True, ret[:len(ret)//2]
    else:
        #logger.warning("solver error")
        return False, None

#@profile
def get_sol_linf(target_x, tuple_x, faropp, kdtree, transformer,
        glob_trnX, glob_trny, init_x=None, n_jobs=1):
    tuple_x = np.asarray(tuple_x)
    fet_dim = target_x.shape[0]
    #n_emb = transformer.shape[0]

    emb_tar = target_x.dot(transformer.T)
    trnX = glob_trnX.dot(transformer.T)
    G, h, _ = get_constraints(trnX, tuple_x, kdtree, faropp, emb_tar)
    G = np.dot(G, transformer)

    if init_x is None and not sol_sat_constraints(G, h):
        return False, None

    c = np.concatenate((np.zeros(fet_dim), np.ones(1))).reshape((-1, 1))

    G2 = np.hstack((np.eye(fet_dim), -np.ones((fet_dim, 1))))
    G3 = np.hstack((-np.eye(fet_dim), -np.ones((fet_dim, 1))))
    G = np.hstack((G, np.zeros((G.shape[0], 1))))
    G = np.vstack((G, G2, G3))
    h = np.concatenate((h, target_x, -target_x)).reshape((-1, 1))

    temph = h - CONSTRAINTTOL

    status, sol = solve_lp(c=c, G=G, h=temph, n=len(c), n_jobs=n_jobs)
    if status == 'optimal':
        ret = np.array(sol).reshape(-1)
        return True, ret[:-1]
    else:
        #print(status)
        #status, sol = solve_lp(c=c, G=G, h=h, n=len(c))
        #print(status)
        #logger.warning("solver error")
        return False, None

    #c, G, h = matrix(c, tc='d'), matrix(G, tc='d'), matrix(h, tc='d')
    #if init_x is not None:
    #    sol = solvers.lp(c=c, G=G, h=temph, solver='glpk',
    #                     initvals=init_x)
    #else:
    #    sol = solvers.lp(c=c, G=G, h=temph, solver='glpk')
    #if sol['status'] == 'optimal':
    #    ret = np.array(sol['x']).reshape(-1)
    #    ### sanity check for the correctness of objective
    #    if DEBUG:
    #        # check if constraints are all met
    #        h = np.array(h).flatten()
    #        G = np.array(G)
    #        a = check_feasibility(G, h, ret, G.shape[0], G.shape[1])
    #        print(a)
    #        print('1', sol['primal objective'])
    #        print('2', np.linalg.norm(target_x - ret[:-1], ord=np.inf))
    #        print('3', np.linalg.norm(target_x.dot(transformer.T) - (ret[:-1]).dot(transformer.T), ord=np.inf))
    #        #print(target_x.dot(transformer.T))
    #        #print(ret)
    #        #assert np.isclose(np.linalg.norm(target_x.dot(transformer.T) - (ret[:-1]).dot(transformer.T), ord=np.inf),
    #        #                  sol['primal objective'], rtol=CONSTRAINTTOL)
    #    return True, ret[:-1]
    #else:
    #    #logger.warning("solver error")
    #    return False, None


#@profile
def get_adv(target_x, target_y, kdtree, farthest, n_neighbors, faropp,
        transformer, lp_sols, ord=2, n_jobs=1):
    ind = kdtree.query(target_x.dot(transformer.T).reshape((1, -1)),
                       k=n_neighbors, return_distance=False)[0]
    if target_y != np.argmax(np.bincount(glob_trny[ind])):
        # already incorrectly predicted
        return np.zeros_like(target_x)

    temp = (target_x, np.inf)
    if farthest == -1:
        farthest = glob_trnX.shape[0]
        ind = np.arange(glob_trnX.shape[0])
    else:
        ind = kdtree.query(target_x.dot(transformer.T).reshape((1, -1)),
                        k=farthest, return_distance=False)
        ind = ind[0]

    combs = []
    for comb in itertools.combinations(range(farthest), n_neighbors):
        comb = list(comb)
        # majority
        if target_y != np.argmax(np.bincount(glob_trny[ind[comb]])):
            combs.append(comb)

    #knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    #knn.fit(glob_trnX.dot(transformer.T), glob_trny)

    if ord == 1:
        get_sol_fn = get_sol_l1
    elif ord == 2:
        get_sol_fn = get_sol
    elif ord == np.inf:
        get_sol_fn = get_sol_linf
    else:
        raise ValueError("Unsupported ord %d" % ord)

    def _helper(comb, transformer, trnX, trny):
        comb_tup = tuple(ind[comb])
        ret, sol = get_sol_fn(target_x, ind[comb], faropp, kdtree,
                              transformer, trnX, trny, n_jobs=n_jobs)
        return ret, sol
    not_vacum = lambda x: tuple(ind[x]) not in lp_sols or lp_sols[tuple(ind[x])]
    combs = list(filter(not_vacum, combs))
    print(len(combs))
    sols = Parallel(n_jobs=7, verbose=1)(
            delayed(_helper)(comb, transformer, glob_trnX, glob_trny) for comb in combs)
    status, sols = zip(*sols)
    for i, s in enumerate(status):
        if not s:
            lp_sols[tuple(ind[combs[i]])] = None

    _, sols = list(zip(*list(filter(lambda s: True if s[0] else False, zip(status, sols)))))
    sols = np.array(list(filter(lambda x: np.linalg.norm(x) != 0, sols)))
    eps = np.linalg.norm(sols - target_x, axis=1, ord=ord)
    temp = (sols[eps.argmin()], eps.min())

    #for comb in combs:
    #    comb_tup = tuple(ind[comb])

    #    if comb_tup not in lp_sols:
    #        ret, sol = get_sol_fn(target_x, ind[comb], faropp, kdtree,
    #                              transformer, glob_trnX, glob_trny, )
    #        lp_sols[comb_tup] = sol
    #    elif lp_sols[comb_tup] is None:
    #        ret = False
    #    else:
    #        ret, sol = get_sol_fn(target_x, ind[comb], faropp, kdtree,
    #                              transformer,glob_trnX, glob_trny,  lp_sols[comb_tup])

    #    if ret:
    #        if knn.predict(sol.reshape(1, -1).dot(transformer.T))[0] == target_y:
    #            print("shouldn't happend")
    #            assert False
    #        else:
    #            eps = np.linalg.norm(sol - target_x, ord=ord)
    #            if eps < temp[1]:
    #                temp = (sol, eps)

    #        if DEBUG:
    #            a = knn.predict(np.dot(target_x.reshape(1, -1), transformer.T))[0]
    #            b = knn.predict(np.dot(sol.reshape(1, -1), transformer.T))[0]
    #            print(a, b, target_y)
    #            if a == b and a == target_y:
    #                print("shouldn't happend")
    #                assert False
    #            #get_sol(target_x, ind[comb], faropp, kdtree, transformer)

    return temp[0] - target_x

def attack_with_eps_constraint(perts, ord, eps):
    perts = np.asarray(perts)
    if isinstance(eps, list):
        rret = []
        norms = np.linalg.norm(perts, axis=1, ord=ord)
        for ep in eps:
            t = np.copy(perts)
            t[norms > ep, :] = 0
            rret.append(t)
        return rret
    elif eps is not None:
        perts[np.linalg.norm(perts, axis=1, ord=ord) > eps, :] = 0
        return perts
    else:
        return perts

#@profile
def rev_get_adv(target_x, target_y, kdtree, farthest, n_neighbors, faropp,
        transformer, lp_sols, ord=2, method='self', knn=None, n_jobs=1):
    if farthest == -1:
        farthest = glob_trnX.shape[0]
    temp = (target_x, np.inf)

    # already predicted wrong
    if knn.predict(target_x.dot(transformer.T).reshape((1, -1)))[0] != target_y:
        return temp[0] - target_x

    if ord == 1:
        get_sol_fn = get_sol_l1
    elif ord == 2:
        get_sol_fn = get_sol
    elif ord == np.inf:
        get_sol_fn = get_sol_linf
    else:
        raise ValueError("Unsupported ord %d" % ord)

    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn.fit(glob_trnX.dot(transformer.T), glob_trny)
    pred_trny = knn.predict(glob_trnX.dot(transformer.T))

    ind = kdtree.query(target_x.dot(transformer.T).reshape((1, -1)),
                       k=len(glob_trnX), return_distance=False)[0]
    ind = list(filter(lambda x: pred_trny[x] != target_y, ind))[:farthest]

    for i in ind:
        if method == 'self':
            inds = [i]
        elif method == 'region':
            procedX = glob_trnX[i].dot(transformer.T).reshape((1, -1))
            inds = kdtree.query(procedX, k=n_neighbors, return_distance=False)[0]
        inds = tuple([_ for _ in inds])

        ret, sol = get_sol_fn(target_x, inds, faropp, kdtree, transformer,
                glob_trnX, glob_trny, init_x=glob_trnX[i], n_jobs=n_jobs)

        if method == 'region':
            #assert ret
            if not ret:
                proc = np.array([glob_trnX[i]]).dot(transformer.T)
                sol = np.array(glob_trnX[i])
            else:
                proc = np.array([sol]).dot(transformer.T)
            assert knn.predict(proc)[0] != target_y
            eps = np.linalg.norm(sol - target_x, ord=ord)
            if eps < temp[1]:
                temp = (sol, eps)
        elif ret: # method == 'self'
            proc = np.array([sol]).dot(transformer.T)
            if knn.predict(proc)[0] != target_y:
                eps = np.linalg.norm(sol - target_x, ord=ord)
                if eps < temp[1]:
                    temp = (sol, eps)

        #if inds not in lp_sols:
        #    ret, sol = get_sol_fn(target_x, inds, faropp, kdtree, transformer)
        #    lp_sols[inds] = sol
        #elif lp_sols[inds] is None:
        #    ret = False
        #    #ret, sol = get_sol_fn(target_x, inds, faropp, kdtree,
        #    #                        transformer)
        #else:
        #    ret, sol = get_sol_fn(target_x, inds, faropp, kdtree,
        #                            transformer, lp_sols[inds])

    return temp[0] - target_x

class NNOptAttack():
    def __init__(self, trnX, trny, n_neighbors=3, farthest=-1, faropp=-1,
            transformer=None, ord=2, n_jobs=1):
        #furthest >= K
        self.n_jobs = n_jobs
        self.K = n_neighbors
        self.trnX = trnX
        self.trny = trny
        self.farthest = min(farthest, len(trnX))
        self.faropp = faropp
        self.transformer = transformer
        self.ord = ord
        if transformer is not None:
            self.tree = KDTree(self.transformer.transform(self.trnX))
        else:
            self.tree = KDTree(self.trnX)
        self.lp_sols = {}
        print(np.shape(self.trnX), len(self.trny))

    def perturb(self, X, y, eps=None, logging=False, n_jobs=1):
        raise NotImplementedError()


class NNAttack(NNOptAttack):
    def __init__(self, trnX, trny, n_neighbors=3, farthest=-1, faropp=-1,
            transformer=None, ord=2, n_jobs=1):
        super().__init__(trnX=trnX, trny=trny, n_neighbors=n_neighbors,
                farthest=farthest, faropp=faropp, transformer=transformer,
                ord=ord, n_jobs=n_jobs)

    #@profile
    def perturb(self, X, y, eps=None, n_jobs=1):
        if self.transformer:
            transformer = self.transformer.transformer()
        else:
            transformer = np.eye(self.trnX.shape[1])

        global glob_trnX
        global glob_trny
        glob_trnX = self.trnX
        glob_trny = self.trny

        #ret = Parallel(n_jobs=-1, backend="threading", batch_size='auto',
        #               verbose=5)(
        #    delayed(get_adv)(target_x, target_y, self.tree, self.farthest, self.K,
        #                     self.faropp, transformer, self.lp_sols,
        #                     ord=self.ord) for target_x, target_y in zip(X, y))

        #knn = KNeighborsClassifier(n_neighbors=self.K)
        #knn.fit(glob_trnX.dot(transformer.T), glob_trny)

        ret = []
        with IterTimer("Perturbing", len(X)) as timer:
            for i, (target_x, target_y) in enumerate(zip(X, y)):
                timer.update(i)
                #ret.append(get_adv(target_x, target_y, self.tree,
                ret.append(get_adv(target_x.astype(np.float64), target_y, self.tree,
                                   self.farthest, self.K, self.faropp,
                                   transformer, self.lp_sols, ord=self.ord))

        self.perts = np.asarray(ret)
        return attack_with_eps_constraint(self.perts, self.ord, eps)


class RevNNAttack(NNOptAttack):
    def __init__(self, trnX, trny, n_neighbors=3, farthest=-1, faropp=-1,
            transformer=None, ord=2, method='self', n_jobs=1):
        super().__init__(trnX=trnX, trny=trny, n_neighbors=n_neighbors,
                farthest=farthest, faropp=faropp, transformer=transformer,
                ord=ord)
        print(method)
        self.method = method

    #@profile
    def perturb(self, X, y, eps=None, n_jobs=1):
        if self.transformer:
            transformer = self.transformer.transformer()
        else:
            transformer = np.eye(self.trnX.shape[1])

        global glob_trnX
        global glob_trny
        glob_trnX = self.trnX
        glob_trny = self.trny

        knn = KNeighborsClassifier(n_neighbors=self.K)
        knn.fit(glob_trnX.dot(transformer.T), glob_trny)

        ret = []
        with IterTimer("Perturbing", len(X)) as timer:
            for i, (target_x, target_y) in enumerate(zip(X, y)):
                timer.update(i)
                ret.append(
                    rev_get_adv(target_x.astype(np.float64), target_y,
                        self.tree, self.farthest, self.K, self.faropp,
                        transformer, self.lp_sols, ord=self.ord,
                        method=self.method, knn=knn, n_jobs=self.n_jobs
                    )
                )

        self.perts = np.asarray(ret)
        return attack_with_eps_constraint(self.perts, self.ord, eps)

class HybridNNAttack(NNOptAttack):
    def __init__(self, trnX, trny, n_neighbors=3, farthest=-1, faropp=-1,
            rev_farthest=-1, transformer=None, ord=2, method='self', n_jobs=1):
        super().__init__(trnX=trnX, trny=trny, n_neighbors=n_neighbors,
                farthest=farthest, faropp=faropp, transformer=transformer,
                ord=ord)
        self.method = method
        self.rev_farthest = rev_farthest

    #@profile
    def perturb(self, X, y, eps=None, n_jobs=1):
        if self.transformer:
            transformer = self.transformer.transformer()
        else:
            transformer = np.eye(self.trnX.shape[1])

        global glob_trnX
        global glob_trny
        glob_trnX = self.trnX
        glob_trny = self.trny

        knn = KNeighborsClassifier(n_neighbors=self.K)
        knn.fit(glob_trnX.dot(transformer.T), glob_trny)

        ret = []
        perts = []
        rev_perts = []
        with IterTimer("Perturbing", len(X)) as timer:
            for i, (target_x, target_y) in enumerate(zip(X, y)):
                timer.update(i)
                pert = get_adv(target_x.astype(np.float64), target_y, self.tree,
                        self.farthest, self.K, self.faropp, transformer,
                        self.lp_sols, ord=self.ord, n_jobs=self.n_jobs)
                rev_pert = rev_get_adv(target_x.astype(np.float64), target_y,
                        self.tree, self.rev_farthest, self.K, self.faropp, transformer,
                        self.lp_sols, ord=self.ord, method=self.method, knn=knn)
                n_pert = np.linalg.norm(pert, ord=self.ord)
                n_revpert = np.linalg.norm(rev_pert, ord=self.ord)
                if n_pert == 0 or (n_pert > n_revpert):
                    ret.append(rev_pert)
                else:
                    ret.append(pert)


        self.perts = np.asarray(ret)
        return attack_with_eps_constraint(self.perts, self.ord, eps)

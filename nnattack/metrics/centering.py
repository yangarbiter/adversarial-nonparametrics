from bistiming import IterTimer
import numpy as np
import scipy
from sklearn.neighbors import KDTree
import cvxopt.glpk
from cvxopt import solvers, spdiag, matrix, sqrt, mul, log, div, blas
from cvxopt.modeling import variable, op
solvers.options['show_progress'] = False
solvers.options['maxiters'] = 30
solvers.options['feastol'] = 1e-2
solvers.options['abstol'] = 1e-2
solvers.options['reltol'] = 1e-2
cvxopt.glpk.options["msg_lev"] = "GLP_MSG_OFF"

from ..attacks.cutils import get_constraints


#@profile
def dataset_centers(X, center_type:str, faropp=-1):
    kdtree = KDTree(X)
    ret = []
    retK = []
    with IterTimer("Calculating centers", len(X)) as timer:
        for i, target_x in enumerate(X):
            G, h, _ = get_constraints(X, np.array([i]), kdtree, faropp, target_x)
            h = h - 1e-5
            K = None
            if center_type == 'max_volume':
                #r, K, c = max_volume_fix_center(G, h, target_x, X)
                r, K, c = max_volume_center(G, h, target_x, X)
                K = scipy.linalg.sqrtm(K)
                #import ipdb; ipdb.set_trace()
                #print(K)
                #print(np.linalg.det(K))
                #assert np.all(np.linalg.eigvals(K) > 0) # K is psd
                #r, K, c = max_volume_center(G, h, target_x, X)
                #print(r)
            elif center_type == 'analytic':
                r, _, c = analytic_center(G, h, target_x)
                K = np.eye(len(target_x))
            elif center_type == 'chebyshev':
                r, _, c = chebyshev_center(G, h, target_x)
                K = np.eye(len(target_x))
            else:
                raise ValueError("not supported center %s", center_type)
            if r not in ['optimal', 'unknown']:
                ret.append(target_x)
                retK.append(np.eye(len(target_x)))
            else:
                ret.append(c.reshape(-1))
                retK.append(np.array(K))
            timer.update(i)
    
    return np.asarray(ret), np.array(retK)

def chebyshev_center(G, h, tar_x):
    # Chebyshev center
    #
    # maximizse   R
    # subject to  gk'*xc + R*||gk||_2 <= hk,  k=1,...,m
    #             R >= 0
    m = G.shape[0]
    n_fets = G.shape[1]

    G = np.hstack((np.linalg.norm(G, ord=2, axis=1, keepdims=True), G))
    G = np.vstack((G, np.array([-1.0] + [0.] * n_fets)))
    h = np.concatenate((h, [0]))
    G, h = matrix(G, tc='d'), matrix(h, tc='d')
    c = matrix([-1.0] + [0.] * n_fets, tc='d')
    sol = solvers.lp(c=c, G=G, h=h, solver='glpk', initvals=matrix(tar_x, tc='d'))

    x = np.array(sol['x']).reshape(-1)

    return sol['status'], None, x[1:]

def max_volume_fix_center(G, h, tar_x, X):
    m = np.shape(G)[0]
    n_fets = np.shape(G)[1]
    x_size = n_fets*(1+n_fets)//2

    diagonal_mask = []
    c = 0
    for i in range(n_fets):
        diagonal_mask.append(int(c+i))
        c += i+1

    D = np.zeros((m, n_fets+1, x_size+n_fets), dtype=np.float64)
    for k in range(m):
        c = 0
        for i in range(n_fets):
            D[k][i][c:c+n_fets-i] = G[k][i:]
            c += n_fets-i
        D[k][-1][-n_fets:] = -G[k]
    d = [([0.0] * n_fets + [float(hk)]) for hk in h]

    G, h = matrix(G, tc='d'), matrix(h, tc='d')
    D = [matrix(Dk, tc='d') for Dk in D]
    d = [matrix(dk, tc='d') for dk in d]

    #@profile
    def F(x=None, z=None):
        if x is None:
            #return m, matrix([ 1.0, 0.0, 1.0, 0.0, 0.0 ])
            ret = np.zeros((x_size, 1), dtype=np.float)
            ret[diagonal_mask, 0] = 1.
            #ret[-n_fets:, 0] = tar_x
            return m, matrix(ret)
        if min(x[diagonal_mask, 0]) <= 0.0:
            return None

        _temp = matrix(np.concatenate((np.array(x), tar_x.reshape(-1, 1))), tc='d')
        y = [Dk*_temp + dk for Dk, dk in zip(D, d)]

        f = np.zeros((m+1, 1))
        for i in diagonal_mask:
            f[0][0] += -log(x[int(i)])
        f = matrix(f, tc='d')

        for k in range(m):
            f[k+1] = y[k][:n_fets].T * y[k][:n_fets] / y[k][n_fets] - y[k][n_fets]

        Df = matrix(0.0, (m+1, x_size))
        for i in diagonal_mask:
            Df[0, i] = -1.0 / x[i]

        # gradient of g is ( 2.0*(u/t);  -(u/t)'*(u/t) -1)
        for k in range(m):
            a = y[k][:n_fets] / y[k][n_fets]
            gradg = matrix(0.0, (n_fets+1,1))
            gradg[:n_fets], gradg[n_fets] = 2.0 * a, -a.T*a - 1
            Df[k+1,:] = (gradg.T * D[k])[:, :x_size]
        if z is None:
            return f, Df

        H = matrix(0.0, (x_size, x_size))
        for i in diagonal_mask:
            H[i,i] = z[0] / x[i]**2

        # Hessian of g is (2.0/t) * [ I, -u/t;  -(u/t)',  (u/t)*(u/t)' ]
        for k in range(m):
            a = y[k][:n_fets] / y[k][n_fets]
            #hessg = matrix(0.0, (n_fets+1, n_fets+1))
            #for i in range(n_fets):
            #    hessg[i, i] = 1.
            hessg = matrix(np.eye(n_fets+1), tc='d')
            hessg[:n_fets, n_fets], hessg[n_fets, :n_fets] = -a, -a.T
            hessg[n_fets, n_fets] = a.T * a
            H += (z[k] * 2.0 / y[k][n_fets]) * (D[k][:, :x_size].T * hessg * D[k][:, :x_size])

        return f, Df, H

    GG = matrix(np.vstack((np.eye(x_size), -np.eye(x_size))), tc='d')
    hh = np.ones(2*x_size)
    hh[:n_fets*(1+n_fets)//2] = (np.max(X) - np.min(X))
    hh[x_size:] = -(np.max(X) - np.min(X))
    hh = matrix(hh, tc='d')

    sol = solvers.cp(F, G=GG, h=hh)
    L = np.array(sol['x'])
    B = np.zeros((n_fets, n_fets))
    for i in range(n_fets):
        if i == 0:
            B[i][:i+1] = L[:i+1, 0]
        else:
            B[i][:i+1] = L[diagonal_mask[i-1]+1:(diagonal_mask[i]+1), 0]
    B = B.dot(B.T).T
    c = np.array(tar_x)
    return sol['status'], B, c

#@profile
def max_volume_center(G, h, tar_x, X):
    # Maximum volume enclosed ellipsoid center
    #
    # minimize    -log det B
    # subject to  ||B * gk||_2 + gk'*c <= hk,  k=1,...,m
    #
    # with variables  B and c.
    #
    # minimize    -log det L
    # subject to  ||L' * gk||_2^2 / (hk - gk'*c) <= hk - gk'*c,  k=1,...,m
    #
    # L lower triangular with positive diagonal and B*B = L*L'.
    #
    # minimize    -log x[0] - log x[2]
    # subject to   g( Dk*x + dk ) <= 0,  k=1,...,m
    #
    # g(u,t) = u'*u/t - t
    # Dk = [ G[k,0]   G[k,1]  0       0        0
    #        0        0       G[k,1]  0        0
    #        0        0       0      -G[k,0]  -G[k,1] ]
    # dk = [0; 0; h[k]]
    #
    # 5 variables x = (L[0,0], L[1,0], L[1,1], c[0], c[1])
    m = np.shape(G)[0]
    n_fets = np.shape(G)[1]
    x_size = n_fets*(1+n_fets)//2 + n_fets

    diagonal_mask = []
    c = 0
    for i in range(n_fets):
        diagonal_mask.append(int(c+i))
        c += i+1

    D = np.zeros((m, n_fets+1, x_size), dtype=np.float64)
    for k in range(m):
        c = 0
        for i in range(n_fets):
            D[k][i][c:c+n_fets-i] = G[k][i:]
            c += n_fets-i
        D[k][-1][-n_fets:] = -G[k]
    d = [([0.0] * n_fets + [float(hk)]) for hk in h]

    G, h = matrix(G, tc='d'), matrix(h, tc='d')
    D = [matrix(Dk, tc='d') for Dk in D]
    d = [matrix(dk, tc='d') for dk in d]

    #@profile
    def F(x=None, z=None):
        if x is None:
            #return m, matrix([ 1.0, 0.0, 1.0, 0.0, 0.0 ])
            ret = np.zeros((x_size, 1), dtype=np.float)
            ret[diagonal_mask, 0] = 1.
            ret[-n_fets:, 0] = tar_x
            return m, matrix(ret)
        if min(min(x[diagonal_mask, 0]), min(h-G*x[-n_fets:])) <= 0.0:
            return None
        #center = np.array(x[-n_fets:, 0])
        #if np.linalg.norm(tar_x-center, ord=2) > 0.1:
        #    return None

        y = [Dk*x + dk for Dk, dk in zip(D, d)]

        f = np.zeros((m+1, 1))
        #f = matrix(0.0, (m+1,1))
        #-log(x[0]) - log(x[2])
        for i in diagonal_mask:
            f[0][0] += -log(x[int(i)])
        f = matrix(f, tc='d')

        for k in range(m):
            f[k+1] = y[k][:n_fets].T * y[k][:n_fets] / y[k][n_fets] - y[k][n_fets]

        Df = matrix(0.0, (m+1, x_size))
        for i in diagonal_mask:
            Df[0, i] = -1.0/x[i]

        # gradient of g is ( 2.0*(u/t);  -(u/t)'*(u/t) -1)
        for k in range(m):
            a = y[k][:n_fets] / y[k][n_fets]
            gradg = matrix(0.0, (n_fets+1,1))
            gradg[:n_fets], gradg[n_fets] = 2.0 * a, -a.T*a - 1
            Df[k+1,:] =  gradg.T * D[k]
        if z is None:
            return f, Df

        H = matrix(0.0, (x_size, x_size))
        for i in diagonal_mask:
            H[i,i] = z[0] / x[i]**2
        #H[0,0] = z[0] / x[0]**2
        #H[2,2] = z[0] / x[2]**2

        # Hessian of g is (2.0/t) * [ I, -u/t;  -(u/t)',  (u/t)*(u/t)' ]
        for k in range(m):
            a = y[k][:n_fets] / y[k][n_fets]
            #hessg = matrix(0.0, (n_fets+1, n_fets+1))
            #for i in range(n_fets):
            #    hessg[i, i] = 1.
            hessg = matrix(np.eye(n_fets+1), (n_fets+1, n_fets+1))
            hessg[:n_fets, n_fets], hessg[n_fets, :n_fets] = -a, -a.T
            hessg[n_fets, n_fets] = a.T * a
            H += (z[k] * 2.0 / y[k][n_fets]) *  D[k].T * hessg * D[k]

        return f, Df, H

    GG = matrix(np.vstack((np.eye(x_size), -np.eye(x_size))), tc='d')
    hh = np.ones(2*x_size)
    hh[:n_fets*(1+n_fets)//2] = (np.max(X) - np.min(X))
    hh[x_size:x_size+n_fets*(1+n_fets)//2] = (np.max(X) - np.min(X))
    hh[n_fets*(1+n_fets)//2:] = np.max(X)
    hh[x_size+n_fets*(1+n_fets)//2:] = -np.min(X)
    hh = matrix(hh, tc='d')

    #import ipdb; ipdb.set_trace()

    sol = solvers.cp(F, G=GG, h=hh)
    L = np.array(sol['x'][:-n_fets])
    B = np.zeros((n_fets, n_fets))
    for i in range(n_fets):
        if i == 0:
            B[i][:i+1] = L[:i+1, 0]
        else:
            B[i][:i+1] = L[diagonal_mask[i-1]+1:(diagonal_mask[i]+1), 0]
    B = B.dot(B.T).T
    c = np.array(sol['x'][-n_fets:])
    return sol['status'], B, c


def analytic_center(G, h, tar_x):
    m = G.shape[0]
    n_fets = G.shape[1]
    G, h = matrix(G, tc='d'), matrix(h, tc='d')

    def F(x=None, z=None):
        if x is None:
            return n_fets, matrix(tar_x, (n_fets,1))
        y = h-G*x
        if min(y) <= 0: return None
        f = -sum(log(y))
        Df = (y**-1).T * G
        if z is None:
            return matrix(f), Df
        H =  G.T * spdiag(y**-2) * G
        return matrix(f, tc='d'), Df, z[0]*H

    try:
        sol = solvers.cp(F)
        Hac = np.array(G.T * spdiag((h-G*sol['x'])**-1) * G)
        xac = np.array(sol['x'])
    except:
        sol = {'status': 'unknown'}
        xac = None
        Hac = None

    return sol['status'], Hac, xac 

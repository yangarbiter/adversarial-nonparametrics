import unittest

from cvxopt import blas, lapack, solvers
from cvxopt import matrix, sqrt, mul, cos, sin, log
from cvxopt.modeling import variable, op
solvers.options['show_progress'] = False
import numpy as np
from numpy.testing import assert_almost_equal

from nnattack.metrics.centering import (
    max_volume_center,
    analytic_center,
    chebyshev_center,
)


def gen_data():
    # Extreme points (with first one appended at the end)
    X = matrix([0.55,  0.25, -0.20, -0.25,  0.00,  0.40,  0.55,
                0.00,  0.35,  0.20, -0.10, -0.30, -0.20,  0.00], (7,2))
    m = X.size[0] - 1

    # Inequality description G*x <= h with h = 1
    G, h = matrix(0.0, (m,2)), matrix(0.0, (m,1))
    G = (X[:m,:] - X[1:, :]) * matrix([0., -1., 1., 0.], (2,2))
    h = (G * X.T)[::m+1]
    G = mul(h[:, [0,0]]**-1, G)
    h = matrix(1.0, (m, 1))

    return np.array(G), np.array(h).reshape(-1)

class TestCentering(unittest.TestCase):
    def setUp(self):
        pass

    def test_max_volume(self):
        G, h = gen_data()
        cvx_r, cvx_L, cvx_c = cvxopt_max_volume(np.array(G), np.array(h))
        r, L, c = max_volume_center(np.array(G), np.array(h), np.array([0, 0]))

        self.assertEqual(cvx_r, 'optimal')
        self.assertEqual(r, 'optimal')
        assert_almost_equal(L, cvx_L, decimal=4)
        assert_almost_equal(c, cvx_c, decimal=4)

    def test_analytic(self):
        G, h = gen_data()
        r, L, c = analytic_center(np.array(G), np.array(h), np.array([0, 0]))

        self.assertEqual(r, 'optimal')
        #assert_almost_equal(L, cvx_L)
        #assert_almost_equal(c, cvx_c)

    def test_chebyshev(self):
        G, h = gen_data()

        r, _, c = chebyshev_center(G, h, np.array([0, 0]))

        m, n_fets = G.shape[0], G.shape[1]
        G, h = matrix(G, tc='d'), matrix(h, tc='d')

        R = variable()
        xc = variable(n_fets)
        op(-R, [G[k,:]*xc + R*blas.nrm2(G[k,:]) <= h[k] for k in range(m)] +
            [R >= 0] ).solve()
        R = R.value
        xc = np.array(xc.value).reshape(-1)

        self.assertEqual(r, 'optimal')
        assert_almost_equal(c, xc, decimal=4)


def cvxopt_max_volume(G, h):
    G, h = matrix(G, tc='d'), matrix(h, tc='d')

    m = len(h)
    D = [matrix(0.0, (3,5)) for k in range(m)]
    for k in range(m):
        D[k][ [0, 3, 7, 11, 14] ] = matrix( [G[k,0], G[k,1], G[k,1],
            -G[k,0], -G[k,1]] )
    d = [matrix([0.0, 0.0, float(hk)]) for hk in h]

    def F(x=None, z=None):
        if x is None:
            return m, matrix([ 1.0, 0.0, 1.0, 0.0, 0.0 ])
        if min(x[0], x[2], min(h-G*x[3:])) <= 0.0:
            return None

        y = [ Dk*x + dk for Dk, dk in zip(D, d) ]

        f = matrix(0.0, (m+1,1))
        f[0] = -log(x[0]) - log(x[2])
        for k in range(m):
            f[k+1] = y[k][:2].T * y[k][:2] / y[k][2] - y[k][2]

        Df = matrix(0.0, (m+1,5))
        Df[0,0], Df[0,2] = -1.0/x[0], -1.0/x[2]

        # gradient of g is ( 2.0*(u/t);  -(u/t)'*(u/t) -1)
        for k in range(m):
            a = y[k][:2] / y[k][2]
            gradg = matrix(0.0, (3,1))
            gradg[:2], gradg[2] = 2.0 * a, -a.T*a - 1
            Df[k+1,:] =  gradg.T * D[k]
        if z is None: return f, Df

        H = matrix(0.0, (5,5))
        H[0,0] = z[0] / x[0]**2
        H[2,2] = z[0] / x[2]**2

        # Hessian of g is (2.0/t) * [ I, -u/t;  -(u/t)',  (u/t)*(u/t)' ]
        for k in range(m):
            a = y[k][:2] / y[k][2]
            hessg = matrix(0.0, (3,3))
            hessg[0,0], hessg[1,1] = 1.0, 1.0
            hessg[:2,2], hessg[2,:2] = -a,  -a.T
            hessg[2, 2] = a.T*a
            H += (z[k] * 2.0 / y[k][2]) *  D[k].T * hessg * D[k]

        return f, Df, H

    sol = solvers.cp(F)
    L = matrix([sol['x'][0], sol['x'][1], 0.0, sol['x'][2]], (2,2))
    c = matrix([sol['x'][3], sol['x'][4]])

    return sol['status'], np.array(L * L.T), np.array(c)

if __name__ == '__main__':
    unittest.main()

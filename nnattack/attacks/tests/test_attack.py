import unittest

import numpy as np
from numpy.testing import assert_almost_equal

from nnattack.attacks.nn_attack import NNAttack


class TestNNAttack(unittest.TestCase):
    def setUp(self):
        pass

    def test_small(self):
        n_neighbors = 3
        trnX = np.array([[0, 1], [1, 0], [1, 1], [2, 2], [-1, -1], [2, -1], [-1, 2]], dtype=np.float)
        trny = np.array([0, 0, 0, 1, 1, 1, 1])
        tstX = np.array([[0, 0]], dtype=np.float)
        tsty = np.array([0])

        attack = NNAttack(trnX, trny, n_neighbors=n_neighbors, faropp=-1,
                          farthest=2*n_neighbors)
        perturb = attack.perturb(tstX, tsty)
        tstX += perturb

        assert_almost_equal(np.array([[-0.5, 0.5]]), perturb, decimal=3)
        # np.array([[0.5, -0.5]]) may also be a solution

if __name__ == '__main__':
    unittest.main()

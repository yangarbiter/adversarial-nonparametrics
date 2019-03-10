import unittest

import numpy as np
import tensorflow as tf
from numpy.testing import assert_almost_equal
from sklearn.metrics import pairwise_distances
from metric_learn import NCA
from keras.datasets import mnist

from nnattack.metrics.nca_tf import NCA_TF

def get_data():
    (X, y), (_, _) = mnist.load_data()
    X = X.reshape(len(X), -1)
    idx1 = np.random.choice(np.where(y==3)[0], 100, replace=False)
    idx2 = np.random.choice(np.where(y==5)[0], 100, replace=False)
    y[idx1] = 0
    y[idx2] = 1
    X = np.vstack((X[idx1], X[idx2])).astype(np.float) / 255.
    y = np.concatenate((y[idx1], y[idx2]))
    return X, y


class TestNCA(unittest.TestCase):
    def setUp(self):
        pass

    def test_nca(self):
        np.random.seed(1126)
        X, y = get_data()
        sess = tf.Session()
        tf.set_random_seed(1126)

        nca_tf = NCA_TF(n_dim=X.shape[1], sess=sess)
        nca_tf.fit(X, y)

        nca = NCA(num_dims=X.shape[1], verbose=0).fit(X, y)
        #print(nca_tf.get_loss(X, y))
        #print(nca_tf.get_loss(X, y, transformer=nca.transformer()))
        # loss -133.00

        t1 = nca_tf.transformer()
        t1 /= t1.max()
        t2 = nca.transformer()
        t2 /= t2.max()
        
        assert_almost_equal(actual=t1, desired=t2, decimal=5)

if __name__ == '__main__':
    unittest.main()
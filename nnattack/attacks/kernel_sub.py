import numpy as np
from sklearn.neighbors import KDTree
from sklearn.preprocessing import OneHotEncoder

class KernelSubModel(object):
    def __init__(self, c, lmnn=None):
        """
        trnX: placeholder
        trny: placeholder, on hot encoded
        """
        self.c = c
        self.enc = OneHotEncoder(sparse=False)
        self.lmnn = lmnn

    def fit(self, X, y):
        if self.lmnn is None:
            self.transformer = np.eye(X.shape[1])
        else:
            self.transformer = self.lmnn.transformer()
        self.trnX = X.dot(self.transformer.T)
        self.trny = y
        self.enc_trny = self.enc.fit_transform(self.trny.reshape(-1, 1))

    def _decision_fn(self, X):
        ret = []
        for i in range(len(X)):
            temp = np.exp(-np.linalg.norm(
                       np.tile(X[i].dot(self.transformer.T), (len(self.trnX), 1)) - self.trnX,
                       axis=1
                   )**2 / self.c).reshape(-1, 1)
            temp = np.sum(temp * self.enc_trny, axis=0)
            ret.append(temp)
        return np.vstack(ret)

    def predict_proba(self, X):
        ret = []
        for i in range(len(X)):
            Xt = np.tile(X[i].dot(self.transformer.T), (len(self.trnX), 1))
            temp = np.exp(-np.linalg.norm(Xt - self.trnX, axis=1)**2 / self.c
                    ).reshape(-1, 1)
            temp = np.sum(temp * self.enc_trny / np.sum(temp), axis=0)
            ret.append(temp)
        return np.vstack(ret)

    def predict(self, X):
        proba = self.predict_proba(X)
        return np.argmax(proba, axis=1)

    def _grad(self, X):
        ret = []
        #for i in range(len(X)):
        #    Xt = np.tile(X[i], (len(self.trnX), 1))
        #    # Xt: (len_trnX, n_fets)
        #    t = np.linalg.norm(Xt - self.trnX, axis=1)
        #    #t = np.sqrt(((Xt-self.trnX)**2).sum(1))
        #    et = np.exp(-t**2 / self.c).reshape(-1, 1)
        #    # et: (len_trnX, 1)
        #    det = -2 * (Xt - self.trnX) / self.c * et
        #    # det: (len_trnX, n_fets)

        #    a = det * np.sum(et)
        #    b = -et * np.sum(det, axis=0)
        #    # b: (len_trnX, n_fets)

        #    #temp = ((a + b) * self.enc_trny).sum(axis=0)
        #    temp = ((a + b) * (1. / np.sum(et))**2).sum(axis=0)
        #    # temp: (n_fets)
        #    ret.append(temp)
        for i in range(len(X)):
            Xt = np.tile(X[i].dot(self.transformer.T), (len(self.trnX), 1))
            et = np.exp(-np.linalg.norm(Xt - self.trnX, axis=1)**2 / self.c).reshape(-1, 1)
            det = -2 * (Xt - self.trnX) / self.c * et
            ret.append(-det.sum(axis=0))
        ret = np.vstack(ret)
        return ret

    def perturb(self, X, y=None, eps=0.1):
        ret = self._grad(X)

        # gradient check
        #ss = 1e-5
        #Xss = np.copy(X)
        #g = self._grad(X)
        ##gm = self._grad(X-ss)
        ##gp = self._grad(X+ss)
        #Xss[:, 0] += ss
        #gp = self._decision_fn(Xss)[:, y[1]]
        #Xss[:, 0] -= 2*ss
        #gm = self._decision_fn(Xss)[:, y[1]]
        #gc = (gp-gm) / 2 / ss
        #print((np.linalg.norm(g[1, 0] - gc[1]) / np.linalg.norm(g[1]) /
        #        np.linalg.norm(gc[1])))
        #print(gp[1], gm[1])
        #print(g[1], gc[1])

        if isinstance(eps, list):
            rret = []
            ret = ret / np.linalg.norm(ret, axis=1).reshape(-1, 1)
            for ep in eps:
                rret.append(ep * ret)
            return rret
        elif eps is not None:
            ret = ret / np.linalg.norm(ret, axis=1).reshape(-1, 1)
            return eps * ret
        else:
            return ret

    #def fprop(self, x, set_ref=False):
    #    r = []
    #    m = tf.shape(self.trnX)[0:1]
    #    i = tf.constant(0)
    #    for i in range(self.n_trn_samples):
    #        temp = tf.exp(-tf.norm(
    #            tf.reshape(tf.tile(x[i], m), [m[0], -1]) - self.trnX, axis=1)**2 / self.c)
    #        temp = temp / tf.reduce_sum(temp)
    #        temp = tf.reduce_sum(temp * self.trny, axis=0)
    #        r.append(temp)
    #    r = tf.stack(r, axis=0)
    #    return r
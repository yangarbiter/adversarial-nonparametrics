import numpy as np
#from keras import backend as K
#from keras.models import Model
#from keras.layers import Dense, Input
#from keras.optimizers import SGD, Nadam, RMSprop, Adadelta, Adam
#from keras import losses
#import tensorflow as tf
from sklearn.neighbors import KDTree
from sklearn.preprocessing import OneHotEncoder
from sklearn.neighbors import KNeighborsClassifier

#from cleverhans.utils_mnist import data_mnist
#from cleverhans.utils_tf import model_train, model_eval
#from cleverhans.attacks import FastGradientMethod
#from cleverhans.model import Model as cleverModel
#from cleverhans.utils import AccuracyReport
#from cleverhans.utils_keras import cnn_model
#from cleverhans.utils_keras import KerasModelWrapper

from .attacks.nn_attack import NNAttack

def get_attack_model(attack_name, trnX, trny, lmnn=None, ord=2, **kwargs):
    if attack_name == 'nnopt':
        n_neighbors = kwargs['n_neighbors']
        attack_model = NNAttack(trnX, trny, n_neighbors=n_neighbors,
                farthest=30, lmnn=lmnn, ord=ord)
    elif attack_name == 'nnopt60':
        n_neighbors = kwargs['n_neighbors']
        attack_model = NNAttack(trnX, trny, n_neighbors=n_neighbors,
                farthest=60, lmnn=lmnn, ord=ord)
    elif attack_name == 'nnopt50_200':
        n_neighbors = kwargs['n_neighbors']
        attack_model = NNAttack(trnX, trny, n_neighbors=n_neighbors,
                farthest=50, lmnn=lmnn, faropp=200, ord=ord)
    elif attack_name == 'kernelSub':
        if lmnn is not None:
            trnX = lmnn.transform(trnX)
        attack_model = KernelSubModel(c=0.1, lmnn=lmnn)
        attack_model.fit(trnX, trny)
    elif attack_name == 'direct':
        n_neighbors = kwargs['n_neighbors']
        attack_model = DirectAttack(n_neighbors=n_neighbors, ord=ord)
        attack_model.fit(trnX, trny)
    elif attack_name == 'fgsm':
        from .attacks.fgsm import KerasFGSM
        attack_model = KerasFGSM(**kwargs)
    else:
        raise ValueError('No such attack:', attack_name)
    return attack_model


def logistic_regression(input_shape, n_classes):
    inputs = Input(shape=input_shape)
    x = Dense(n_classes, activation='sigmoid')(inputs)
    return Model(input=[inputs], output=[x])

class KerasModel():
    def __init__(self, n_features, n_classes, batch_size=128,
                 architecture='logistic_regression', n_epochs=10, learning_rate=1e-2):
        self.input_shape = (batch_size, n_features)
        self.n_epochs = n_epochs
        self.n_classes = n_classes
        self.batch_size = batch_size
        self.model = globals()[architecture]
        optimizer = Adam(lr=learning_rate)

        self.model.compile(loss=losses.categorical_crossentropy, optimizer=optimizer)

    def fit(self, X, y):
        self.model.fit(X, y, epochs=self.n_epochs, batch_size=self.batch_size)

class KernelSubModel():
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

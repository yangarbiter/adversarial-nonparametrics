import numpy as np
from sklearn.preprocessing import OneHotEncoder

import tensorflow as tf
from cleverhans.attacks import ProjectedGradientDescent
from cleverhans.utils_keras import KerasModelWrapper
import keras
from keras import backend
import numpy as np


class KerasPGD(object):
    def __init__(self, n_feats, n_classes, model, sess, lbl_enc, ord):
        self.sess = sess
        self.model = model
        self.ord = ord
        self.x = tf.placeholder(tf.float32, shape=(None, n_feats))
        self.y = tf.placeholder(tf.float32, shape=(None, n_classes))

        self.wrap = KerasModelWrapper(model)
        self.lbl_enc = lbl_enc

    def _get_pert(self, X, Y, eps):
        if eps == 0:
            return np.zeros_like(X)
        with self.sess.as_default():
            self.x = self.wrap.input

            pgd = ProjectedGradientDescent(self.x, sess=self.sess)
            adv_x = pgd.generate(self.x, y=self.y, eps=eps, ord=self.ord, eps_iter=0.01)
            adv_x = tf.stop_gradient(adv_x)
            pert_x = adv_x - self.x

            feed_dict = {self.x: X, self.y: Y}
            ret = pert_x.eval(feed_dict=feed_dict)
        return ret

    def perturb(self, X, y, eps=0.1):
        Y = self.lbl_enc.transform(y.reshape(-1, 1))
        if isinstance(eps, list):
            rret = []
            for ep in eps:
                rret.append(self._get_pert(X, Y, ep))
            return rret
        elif isinstance(eps, float):
            ret = self._get_pert(X, y, eps)
        return ret

from functools import partial

import numpy as np
import tensorflow as tf
from sklearn.preprocessing import OneHotEncoder
from cleverhans.attacks.fast_gradient_method import optimize_linear
from cleverhans import utils_tf
from cleverhans.utils_tf import clip_eta

from .robust_nn.eps_separation import find_eps_separated_set

class KernelSubTFModel(object):
    def __init__(self, c, sess, ord, lbl_enc:OneHotEncoder,
            train_type: str=None, eps:float = 0.1):
        """
        trnX: placeholder
        trny: placeholder, on hot encoded
        """
        self.c = c
        self.sess = sess
        self.enc = lbl_enc
        self.ord = ord
        self.train_type = train_type
        self.eps = eps

    def fit(self, X, y):
        self.trnX, self.trny = X, y
        self.enc_trny = self.enc.transform(self.trny.reshape(-1, 1))
        print("original X", np.shape(X), len(y))

        if self.train_type == 'adv':
            enc_trny = self.enc.transform(y.reshape(-1, 1))
            x = tf.placeholder(tf.float32, shape=X.shape)
            indices = tf.constant([[i, j] for i, j in enumerate(y)], tf.int32)
            def loss_fn(x):
                d_fn = self._decision_fn(
                    x,
                    enc_trny=tf.constant(enc_trny, tf.float32),
                    trn_x=tf.constant(X, tf.float32),
                )
                return tf.gather_nd(-d_fn, indices=indices)
            adv_x = pgd_perturb(x, y=tf.constant(y, tf.int32), loss_fn=loss_fn,
                            ord=self.ord, eps=self.eps)
            adv_X = adv_x.eval(feed_dict={x: X}, session=self.sess)

            self.trnX = np.vstack((X, adv_X))
            self.trny = np.concatenate((y, y))
            self.augX, self.augy = self.trnX, self.trny
            print("number of augX", np.shape(self.augX), len(self.augy))
        elif self.train_type == 'advPruning':
            y = y.astype(int)*2-1
            self.augX, self.augy = find_eps_separated_set(X, self.eps/2, y)
            self.augy = (self.augy+1)//2
            self.trnX, self.trny = self.augX, self.augy
            assert len(self.trnX) == len(self.trny)
            assert len(self.augX) == len(self.augy)
            print("number of augX", np.shape(self.augX), len(self.augy))
        elif self.train_type is None:
            self.trnX, self.trny = X, y
        else:
            raise ValueError("Not supported train type: %s", self.train_type)


        self.enc_trny = self.enc.transform(self.trny.reshape(-1, 1))

    def predict_proba(self, X):
        x = tf.placeholder(tf.float32)

        proba = self._decision_fn(x,
                              enc_trny=tf.constant(self.enc_trny, tf.float32),
                              trn_x=tf.constant(self.trnX, tf.float32))
        ret = proba.eval(feed_dict={x: X}, session=self.sess)

        return ret

    def predict(self, X):
        ret = self.predict_proba(X)

        return np.argmax(ret, axis=1)

    def perturb(self, X, y:np.array=None, eps=0.1):
        x = tf.placeholder(tf.float32, shape=X.shape)
        indices = tf.constant([[i, j] for i, j in enumerate(y)], tf.int32)

        def loss_fn(x):
            d_fn = self._decision_fn(
                x,
                enc_trny=tf.constant(self.enc_trny, tf.float32),
                trn_x=tf.constant(self.trnX, tf.float32),
            )
            return tf.gather_nd(-d_fn, indices=indices)

        #adv_x = pgd_perturb(x, y=tf.constant(y, tf.int32), loss_fn=loss_fn, ord=self.ord)
        #pert = adv_x - x
        #ret = pert.eval(feed_dict={x: X}, session=self.sess)
        #import ipdb; ipdb.set_trace()

        if isinstance(eps, list):
            rret = []
            for ep in eps:
                adv_x = pgd_perturb(x, y=tf.constant(y, tf.int32),
                                    loss_fn=loss_fn, eps=ep, ord=self.ord)
                pert = adv_x - x
                ret = pert.eval(feed_dict={x: X}, session=self.sess)
                rret.append(ret)
            return rret
        elif eps is not None:
            adv_x = pgd_perturb(x, y=tf.constant(y, tf.int32),
                                loss_fn=loss_fn, eps=eps, ord=self.ord)
            pert = adv_x - x
            ret = pert.eval(feed_dict={x: X}, session=self.sess)
            return ret
        else:
            raise ValueError("No EPS provided.")

    def _decision_fn(self, x, enc_trny, trn_x):
        trnX, X = trn_x, x
        #mask = tf.constant(y[:, tf.newaxis] == y[tf.newaxis, :], dtype=tf.float32)

        r1 = tf.reduce_sum(X*X, 1)[:, tf.newaxis]
        r2 = tf.reduce_sum(trnX*trnX, 1)[:, tf.newaxis]
        p_ij = r1 - 2*tf.matmul(X, tf.transpose(trnX)) + tf.transpose(r2) # l2 dist
        p_ij = p_ij / self.c
        p_ij = tf.exp(-p_ij)
        # (n_samples, trn_n_samples)

        proba = tf.reduce_sum(p_ij[:, :, tf.newaxis] * enc_trny, axis=1) \
                / tf.reduce_sum(p_ij, axis=1)[:, tf.newaxis]

        return proba

def fgm_perturb(x, y, loss_fn, clip_min=None, clip_max=None, ord=np.inf, eps=0.3):
    loss = loss_fn(x)
    grad, = tf.gradients(loss, x)
    optimal_perturbation = optimize_linear(grad, eps, ord)
    adv_x = x + optimal_perturbation

    if (clip_min is not None) or (clip_max is not None):
        # We don't currently support one-sided clipping
        assert clip_min is not None and clip_max is not None
        adv_x = utils_tf.clip_by_value(adv_x, clip_min, clip_max)

    return adv_x


def pgd_perturb(x, y, loss_fn, y_target=None, clip_min=None,
                clip_max=None, rand_init=False, ord=np.inf, eps=0.3,
                eps_iter=0.05, rand_minmax=0.3, nb_iter=10):
    # Initialize loop variables
    if rand_init:
        eta = tf.random_uniform(tf.shape(x),
                                tf.cast(-rand_minmax, x.dtype),
                                tf.cast(rand_minmax, x.dtype),
                                dtype=x.dtype)
    else:
        eta = tf.zeros(tf.shape(x))

    # Clip eta
    eta = clip_eta(eta, ord, eps)
    adv_x = x + eta
    if clip_min is not None or clip_max is not None:
        adv_x = utils_tf.clip_by_value(adv_x, clip_min, clip_max)

    if y_target is not None:
        y = y_target
        targeted = True
    elif y is not None:
        y = y
        targeted = False
    else:
        raise ValueError
    #    model_preds = self.model.get_probs(x)
    #    preds_max = reduce_max(model_preds, 1, keepdims=True)
    #    y = tf.to_float(tf.equal(model_preds, preds_max))
    #    y = tf.stop_gradient(y)
    #    targeted = False
    #    del model_preds

    y_kwarg = 'y_target' if targeted else 'y'
    fgm_params = {
        'loss_fn': loss_fn,
        'eps': eps_iter,
        y_kwarg: y,
        'ord': ord,
        'clip_min': clip_min,
        'clip_max': clip_max
    }
    if ord == 1:
        raise NotImplementedError("It's not clear that FGM is a good inner loop"
                                    " step for PGD when ord=1, because ord=1 FGM "
                                    " changes only one pixel at a time. We need "
                                    " to rigorously test a strong ord=1 PGD "
                                    "before enabling this feature.")

    # Use getattr() to avoid errors in eager execution attacks
    #FGM = self.FGM_CLASS(
    #    self.model,
    #    sess=getattr(self, 'sess', None),
    #    dtypestr=self.dtypestr)

    def cond(i, _):
        return tf.less(i, nb_iter)

    def body(i, adv_x):
        adv_x = fgm_perturb(adv_x, **fgm_params)

        # Clipping perturbation eta to self.ord norm ball
        eta = adv_x - x
        eta = clip_eta(eta, ord, eps)
        adv_x = x + eta

        # Redo the clipping.
        # FGM already did it, but subtracting and re-adding eta can add some
        # small numerical error.
        if clip_min is not None or clip_max is not None:
            adv_x = utils_tf.clip_by_value(adv_x, clip_min, clip_max)

        return i + 1, adv_x

    _, adv_x = tf.while_loop(cond, body, (tf.zeros([]), adv_x), back_prop=True,
                            maximum_iterations=nb_iter)

    #if self.sanity_checks:
    #    with tf.control_dependencies(asserts):
    #        adv_x = tf.identity(adv_x)

    return adv_x

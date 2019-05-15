""""""
from functools import partial

import numpy as np
import tensorflow as tf
from tensorflow.python import debug as tf_debug
from cleverhans.attacks import FastGradientMethod, ProjectedGradientDescent
from cleverhans.attacks.fast_gradient_method import optimize_linear
from cleverhans import utils_tf
from cleverhans.utils_tf import clip_eta

EPS = np.finfo(float).eps

class KernelSubTf(object):
    """Kernel substitution attack."""
    def __init__(self, sess, c=1.0, attack:str=None, ord=np.inf):
        self.sess = sess
        self.attack = attack
        self.ord = ord
        if isinstance(c, float):
            self.c = tf.constant(c, dtype=tf.float32)
        else:
            self.c = tf.Variable(1.0, name='c')
            self.c = tf.clip_by_value(self.c, 0, np.infty)

    def fit(self, X, y):
        pass

    def _get_adv_X(self, X, y, eps:float):
        with tf.Session() as sess:
            #x = tf.placeholder(tf.float32, shape=(None, X.shape[1]))
            x = tf.constant(X, tf.float32)
            if self.attack == 'fgsm':
                adv_x = fgm_perturb(x, y=tf.constant(y, tf.float32), eps=eps, ord=self.ord,
                    loss_fn=partial(self._loss_fn, y=y, c=self.c))
            elif self.attack == 'pgd':
                adv_x = pgd_perturb(x, y=tf.constant(y, tf.float32), eps=eps, ord=self.ord,
                    loss_fn=partial(self._loss_fn, y=y, c=self.c))
            else:
                raise ValueError("not supported attack method %s", self.attack)
            #return sess.run(adv_x, feed_dict={x: X})
            return sess.run(adv_x)

    def perturb(self, X, y, eps=0.1):
        if isinstance(eps, list):
            ret = []
            for ep in eps:
                ret.append(self._get_adv_X(X, y, ep) - X)
        elif eps is not None:
            ret = self._get_adv_X(X, y, eps) - X

        return ret

    def _loss_fn(self, x, y, c):
        #X = tf.matmul(x, tf.transpose(transformer))
        X = x
        mask = tf.constant(y[:, tf.newaxis] == y[tf.newaxis, :], dtype=tf.float32)

        r = tf.reduce_sum(X*X, 1)[:, tf.newaxis]
        p_ij = r - 2*tf.matmul(X, tf.transpose(X)) + tf.transpose(r) # l2 dist
        p_ij = p_ij / self.c

        p_ij = tf.linalg.set_diag(p_ij, np.ones(len(y), dtype=np.float32) * np.inf)
        p_ij = tf.exp(-p_ij - tf.reduce_logsumexp(-p_ij, axis=1)[:, tf.newaxis])
        # (n_samples, n_samples)

        # Compute loss
        masked_p_ij = p_ij * mask
        #p = masked_p_ij.sum(axis=1, keepdims=True)  # (n_samples, 1)
        p = tf.reduce_sum(masked_p_ij, axis=1, keepdims=True)  # (n_samples, 1)
        loss = -tf.reduce_sum(p)

        return loss

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
            clip_max=None, rand_init=False, ord=np.inf, eps=0.3, eps_iter=0.1,
            rand_minmax=0.3, nb_iter=20):
    # changed nb_iter to 20 and eps_iter to 0.1 for higher eps attack
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

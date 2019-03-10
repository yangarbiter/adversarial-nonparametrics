from functools import partial

import numpy as np
import tensorflow as tf
from tensorflow.python import debug as tf_debug
from cleverhans.attacks.fast_gradient_method import optimize_linear
from cleverhans import utils_tf
from cleverhans.utils_tf import clip_eta
from sklearn.preprocessing import OneHotEncoder

from .centering import dataset_centers
from ..attacks.nn_attack import RevNNAttack, NNAttack
from ..attacks.direct import DirectAttack

EPS = np.finfo(float).eps

class NCA_TF(object):
    def __init__(self, n_dim, sess, init:str='diagonal', n_inits=1,
                 center_type:str='data', c=1.0, attack:str=None,
                 n_neighbors:int=3, adv_type:str='only', ord=np.inf, eps=0.1):
        """
        trnX: placeholder
        trny: placeholder, on hot encoded
        """
        self.n_dim = n_dim
        self.sess = sess
        #self.sess = tf_debug.LocalCLIDebugWrapperSession(sess)
        self.n_neighbors = n_neighbors
        self.enc = OneHotEncoder(sparse=False)
        self.attack = attack
        self.ord = ord
        self.eps = eps
        self.init = init
        self.n_inits = init
        self.adv_type = adv_type
        self.center_type = center_type
        if isinstance(c, float):
            self.c = tf.constant(c, dtype=tf.float32)
        else:
            self.c = tf.Variable(1.0, name='c')
            self.c = tf.clip_by_value(self.c, 0, np.infty)

    def fit(self, X, y):
        self.trn_x = tf.constant(X, dtype=np.float32)
        trny = y
        self.n_fets = X.shape[1]
        self.enc_trny = self.enc.fit_transform(y.reshape(-1, 1))
        #self.transformer_var = tf.get_variable("transformer",
        #                                   [np.shape(X)[1], self.n_dim],
        #                                   trainable=True)
        if self.init == 'diagonal':
            A = np.zeros([self.n_fets, self.n_dim])
            np.fill_diagonal(A, 1./(np.maximum(X.max(axis=0)-X.min(axis=0), EPS)))
            self.transformer_var = tf.Variable(A, name='transformer_nca', dtype=tf.float32)
            self.n_inits = 1
        elif self.init == 'uniform':
            scaler = 1./(np.maximum(X.max(axis=0)-X.min(axis=0), EPS))
            self.transformer_var = tf.Variable(
                tf.random.uniform([self.n_fets, self.n_dim]) * scaler,
                name='transformer_nca',
                dtype=tf.float32)
        else:
            raise ValueError("init %s not supported", self.init)

        x = tf.placeholder(tf.float32, shape=(None, self.n_fets))
        #loss =  self._loss_fn(x)
        #opt = tf.train.AdamOptimizer(learning_rate=1e-3).minimize(loss=loss, var_list=[self.transformer_var])
        #self.sess.run(tf.global_variables_initializer())
        #self.sess.run(opt, feed_dict={x: X})

        if self.center_type == 'data':
            X_centers, covariance = X, np.array([np.eye(self.n_fets) for _ in range(len(X))])
            loss_fn_fn = self.__loss_fn # uses less memory
        else:
            X_centers, covariance = dataset_centers(X, self.center_type)
            loss_fn_fn = self._loss_fn
        self.covariance = tf.constant(covariance, dtype=np.float32)

        if self.attack == 'fgsm':
            adv_x = fgm_perturb(x,
                y=tf.constant(y, tf.float32), eps=self.eps, ord=self.ord,
                loss_fn=partial(loss_fn_fn, transformer=self.transformer_var, y=y))
        elif self.attack == 'pgd':
            adv_x = pgd_perturb(x,
                y=tf.constant(y, tf.float32), eps=self.eps, ord=self.ord,
                loss_fn=partial(loss_fn_fn, transformer=self.transformer_var, y=y))
        #elif isinstance(self.attack, NNAttack) or isinstance(self.attack, RevNNAttack):
        #    direct = DirectAttack(n_neighbors=self.n_neighbors, ord=self.ord)
        #    direct.fit(X, y)
        #    di_pert = direct.perturb(X, y, eps=self.eps)
        #    pert = self.attack.perturb(X, y, eps=self.eps)
        #    idx = np.where(pert.sum(1)==0)
        #    pert[idx] = di_pert[idx]

        #    _adv_X = X + pert
        #    adv_x = tf.constant(_adv_X, tf.float32)
        elif isinstance(self.attack, NNAttack) \
                or isinstance(self.attack, RevNNAttack) \
                or isinstance(self.attack, DirectAttack):
            _adv_X = X + self.attack.perturb(X, y, eps=self.eps)
            adv_x = tf.constant(_adv_X, tf.float32)
        elif self.attack is None:
            adv_x = x
        else:
            raise ValueError("not supported attack method %s", self.attack)

        adv_x = tf.stop_gradient(adv_x)
        loss = loss_fn_fn(adv_x, self.transformer_var, trny)

        if self.adv_type == 'both':
            loss += loss_fn_fn(x, self.transformer_var, trny)

        var_list = [self.transformer_var]
        if isinstance(self.c, tf.Variable):
            var_list.append(self.c)
        #opt = tf.train.AdamOptimizer(learning_rate=1e-9).minimize(loss=loss, var_list=var_list)
        #opt = tf.train.RMSPropOptimizer(learning_rate=1e-9).minimize(loss=loss, var_list=var_list)
        #opt = tf.train.GradientDescentOptimizer(learning_rate=1e-10).minimize(loss=loss, var_list=var_list)
        self.sess.run(tf.global_variables_initializer())
        #from sklearn.metrics import pairwise_distances
        #X_centers = X_centers.astype(np.float32)
        #aa = X_centers.dot(A.T).astype(np.float32)
        #ttt = (aa*aa).sum(1)  - 2*aa.dot(aa.T) + (aa*aa).sum(1).T

        #aaa = self.sess.run(ddd, {adv_x:X_centers})
        #import ipdb; ipdb.set_trace()
        #ttt = np.tensordot(X_centers.dot(A.T)[:, np.newaxis, :], np.linalg.inv(covariance), axes=((2), (1)))
        #ttt = np.transpose(ttt, [2, 0, 3, 1])[:, :, :, 0]
        #self.sess.run(self._loss_fn(adv_x, self.transformer_var, y, self.c)[0], {adv_x:X_centers})
        #import ipdb; ipdb.set_trace()
        #print(self.sess.run(loss, feed_dict={x: X}))
        #self.sess.run(opt, feed_dict={x: X_centers})
        #print(self.sess.run(loss, feed_dict={x: X}))

        tf.contrib.opt.ScipyOptimizerInterface(
            loss, var_list=var_list,
             options={'maxiter': 100}, method='L-BFGS-B').minimize(
                 self.sess, feed_dict={x: X_centers})

        #from metric_learn import NCA
        #t = NCA().fit(X, y)
        ##print(t.transformer())
        #self.sess.run(tf.assign(self.transformer_var, t.transformer()))
        #print(self.sess.run(loss, feed_dict={x: X}))
        ##print(self.transformer_var.eval(session=self.sess))

    def get_loss(self, X, y, transformer=None):
        if transformer is not None:
            transformer = tf.constant(transformer, dtype=np.float32)
        else:
            transformer = self.transformer_var
        x = tf.placeholder(tf.float32, shape=(None, self.n_fets))
        loss = self._loss_fn(x, transformer, y)
        return self.sess.run(loss, feed_dict={x: X})

    def _loss_fn(self, x, transformer, y):
        trnX = tf.matmul(self.trn_x, tf.transpose(transformer))
        X = tf.matmul(x, tf.transpose(transformer))
        mask = tf.constant(y[:, tf.newaxis] == y[tf.newaxis, :], dtype=tf.float32)
        cov = self.covariance
        inv_cov = tf.linalg.inv(cov)

        def pairwise(A, B):
            p_dist = tf.tensordot(A[:, tf.newaxis], inv_cov, axes=((2), (1)))
            p_dist = tf.transpose(p_dist, perm=[2, 0, 3, 1]) # (n, n, fets, 1)
            p_dist = p_dist[:, :, :, 0]
            def cond(i, _):
                return tf.less(i, len(y))
            def body(i, c):
                _t = tf.matmul(p_dist[i], B[i, :, tf.newaxis])[tf.newaxis, :]
                return i+1, tf.concat((c, _t), axis=0)
            _, p_dist = tf.while_loop(cond, body,
                                (tf.zeros([], dtype=tf.int32), tf.zeros((1, len(y), 1), dtype=tf.float32)),
                                shape_invariants=(tf.TensorShape([]), tf.TensorShape([None, len(y), 1]), ),
                                back_prop=True,
                                maximum_iterations=len(y))
            p_dist = p_dist[1:, :, 0]
            return p_dist

        #r1 = tf.linalg.tensor_diag_part(pairwise(X, X))[:, tf.newaxis]
        r1 = pairwise(X, X)
        r2 = pairwise(X, trnX)
        r3 = pairwise(trnX, trnX)
        #r3 = tf.linalg.tensor_diag_part(pairwise(trnX, trnX))[:, tf.newaxis]

        #r = tf.linalg.tensor_diag_part(p_dist)[:, tf.newaxis]
        p_ij = r1 - 2*r2 + tf.transpose(r3) # l2 dist (n, n)
        p_ij /= 2. # (n, n)
        p_ij /= self.c

        p_ij = tf.linalg.set_diag(p_ij, np.ones(len(y), dtype=np.float32) * np.inf)
        #buttom = tf.log(tf.sqrt(tf.linalg.det(cov))[:, tf.newaxis])
        #p_ij = tf.exp(-p_ij - tf.transpose(tf.tile(buttom, (1, len(y)))) \
        #             - tf.reduce_logsumexp(-p_ij, axis=1)[:, tf.newaxis] + tf.transpose(tf.reduce_sum(buttom)))
        p_ij = tf.exp(-p_ij - tf.reduce_logsumexp(-p_ij, axis=1)[:, tf.newaxis])
        # (n_samples, n_samples)

        # Compute loss
        masked_p_ij = p_ij * mask
        #p = masked_p_ij.sum(axis=1, keepdims=True)  # (n_samples, 1)
        p = tf.reduce_sum(masked_p_ij, axis=1, keepdims=True)  # (n_samples, 1)
        loss = -tf.reduce_sum(p)

        return loss

    def __loss_fn(self, x, transformer, y):
        X = tf.matmul(x, tf.transpose(transformer))
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

    def transform(self, X):
        transformer = self.transformer()
        return X.dot(transformer.T)
        #transform = tf.matmul(tf.constant(X, tf.float32), self.transformer_var)
        #return self.sess.run(transform).astype(np.float64)

    def transformer(self):
        ret = self.transformer_var.eval(session=self.sess)
        return ret.astype(np.float64)

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
            clip_max=None, rand_init=False, ord=np.inf, eps=0.3, eps_iter=0.05,
            rand_minmax=0.3, nb_iter=10):
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

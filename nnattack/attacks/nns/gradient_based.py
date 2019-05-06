"""
"""

import numpy as np
import tensorflow as tf
from sklearn.neighbors import KNeighborsClassifier

from .base import AttackModel

EPS=1e-8


def compute_cosine(u, v):
    """
    Asssume normalized inputs
    """
    assert u.ndim >= v.ndim
    v_rs = v.reshape(-1)
    if u.ndim > v.ndim:
        u_rs = u.reshape(len(u), -1)
        cdist = np.array([1 - np.dot(u_i, v_rs) for u_i in u_rs])
    else:
        u_rs = u.reshape(-1)
        cdist = 1 - np.dot(u_rs, v_rs)
    return cdist

def find_2nd_nn_l2(Q, y_Q, X, y_X, k):
    assert Q.shape[1:] == X.shape[1:]
    nn = np.zeros((len(Q), k), dtype=np.int32)
    axis = tuple(np.arange(1, X.ndim, dtype=np.int32))
    for i, q in enumerate(Q):
        dist = np.sum((X - q)**2, axis=axis)
        ind = np.argsort(dist)
        mean_dist = np.zeros((10,))
        for j in range(10):
            ind_j = ind[y_X[ind] == j]
            dist_j = dist[ind_j][:k]
            mean_dist[j] = np.mean(dist_j)
        ind_dist = np.argsort(mean_dist)
        if ind_dist[0] == y_Q[i]:
            nn[i] = ind[y_X[ind] == ind_dist[1]][:k]
        else:
            nn[i] = ind[y_X[ind] == ind_dist[0]][:k]
    return nn

def find_nn(Q, X, k):
    assert Q.shape[1:] == X.shape[1:]
    nn = np.zeros((len(Q), k), dtype=np.int32)

    Q_nm = Q / np.sqrt(np.sum(Q**2, axis=1, keepdims=True))
    X_nm = X / np.sqrt(np.sum(X**2, axis=1, keepdims=True))

    for i, q in enumerate(Q_nm):
        ind = np.argsort(compute_cosine(X_nm, q))[:k]
        nn[i] = ind
    return nn

#def find_nn(Q, X, k):
#    assert Q.shape[1:] == X.shape[1:]
#    nn = np.zeros((len(Q), k), dtype=np.int32)
#
#    for i, q in enumerate(Q):
#        ind = np.argsort(np.linalg.norm(X-q, axis=1))[:k]
#        nn[i] = ind
#    return nn

def classify(nn, y_X):
    vote = np.array([np.argmax(np.bincount(y)) for y in y_X[nn]])
    return vote

class GradientBased(AttackModel):
    """
    referenced from https://github.com/chawins/dknn_attack/blob/7b29cabd30e552832fc7412d5bbdbb7f7ade997c/lib/knn_attack.py
    """

    def __init__(self, sess, trnX, trny, ord, m=75, batch_size=100,
            abort_early=True, init_const=1, min_dist=1, learning_rate=1e-1,
            bin_search_steps=5, max_iter=300, rnd_start=0):
        super().__init__(ord=ord)
        self.sess = sess
        self.trnX = trnX
        self.trny = trny
        self.batch_size = batch_size
        self.abort_early = abort_early
        self.init_const = init_const
        self.learning_rate = learning_rate
        self.min_dist = min_dist
        self.m = m

        self.bin_search_steps = bin_search_steps
        self.max_iter = max_iter
        self.rnd_start = rnd_start

    def perturb(self, X, y, eps=0.1):
        self.build_graph(.2)
        X_adv = self.attack(X, y, .2)

        X_train_nm = self.trnX / (np.sqrt(np.sum(self.trnX**2, axis=1,
            keepdims=True)) + EPS)
        X_adv_nm = X_adv / (np.sqrt(np.sum(X_adv**2, axis=1, keepdims=True)) +
                EPS)
        nn = find_nn(X_adv_nm, X_train_nm, 1)
        y_knn = classify(nn, self.trny)
        print("score:", np.mean(y_knn == y))
        print(y_knn)
        X_nm = X / (np.sqrt(np.sum(X**2, axis=1, keepdims=True)) + EPS)

        knn = KNeighborsClassifier(n_neighbors=1).fit(X_train_nm, self.trny)
        pred = knn.predict(X_nm)
        pred_adv = knn.predict(X_adv_nm)
        print(np.linalg.norm(X_adv - X_nm, ord=self.ord, axis=1))
        print(np.linalg.norm(X_adv_nm - X_nm, ord=2, axis=1).mean())
        print(knn.score(X, y))
        print(knn.score(X_adv, y))

        import ipdb; ipdb.set_trace()

        pert_X = X_adv - X
        self.perts = pert_X - X

        if isinstance(eps, list):
            rret = []
            norms = np.linalg.norm(pert_X, axis=1, ord=self.ord)
            for ep in eps:
                t = np.copy(pert_X)
                t[norms > ep, :] = 0
                rret.append(t)
            return rret
        elif eps is not None:
            pert_X[np.linalg.norm(pert_X, axis=1, ord=self.ord) > eps, :] = 0
            return pert_X
        else:
            return pert_X

    def attack(self, Q, y_Q, eps):
        r = []
        for i in range(0, len(Q), self.batch_size):
            print("Running Baseline Attack on instance {} of {}".format(i, len(Q)))
            #t1 = timeit.default_timer()
            Q_batch = Q[i:i + self.batch_size]
            y_batch = y_Q[i:i + self.batch_size]
            real_len = Q_batch.shape[0]
            if real_len != self.batch_size:
                pad_Q = ((0, self.batch_size - real_len), (0, 0))
                pad_y = ((0, self.batch_size - real_len))
                Q_batch = np.pad(Q_batch, pad_Q, 'constant')
                y_batch = np.pad(y_batch, pad_y, 'constant')
            r.extend(self.attack_batch(Q_batch, y_batch, eps=eps))
            #t2 = timeit.default_timer()
            #print("Time this batch: {:.0f}s".format(t2 - t1))
        return np.array(r)[:len(Q)]

    def build_graph(self, eps):
        batch_size = self.batch_size
        m = self.m
        min_dist = self.min_dist

        input_ndim = self.trnX.ndim
        input_axis = np.arange(1, input_ndim)
        input_shape = (batch_size, ) + self.trnX.shape[1:]

        # Objective variable
        modifier = tf.Variable(np.zeros(input_shape), dtype=tf.float32)

        # These are variables to be more efficient in sending data to tf
        q_var = tf.Variable(np.zeros(input_shape),
                            dtype=tf.float32, name='q_var')
        x_var = tf.Variable(np.zeros((batch_size, m) + self.trnX.shape[1:]),
                            dtype=tf.float32,
                            name='x_var')
        const_var = tf.Variable(
            np.zeros(batch_size), dtype=tf.float32, name='const_var')
        w_var = tf.Variable(
            np.zeros((batch_size, m, 1)), dtype=tf.float32, name='w_var')
        steep_var = tf.Variable(
            np.zeros((batch_size, 1, 1)), dtype=tf.float32, name='const_var')
        clipmin_var = tf.Variable(
            np.zeros(input_shape), dtype=tf.float32, name='clipmin_var')
        clipmax_var = tf.Variable(
            np.zeros(input_shape), dtype=tf.float32, name='clipmax_var')

        # and here's what we use to assign them
        self.assign_q = tf.placeholder(
            tf.float32, input_shape, name='assign_q')
        self.assign_x = tf.placeholder(
            tf.float32, (batch_size, m) + self.trnX.shape[1:], name='assign_x')
        self.assign_const = tf.placeholder(
            tf.float32, [batch_size], name='assign_const')
        self.assign_w = tf.placeholder(
            tf.float32, [batch_size, m, 1], name='assign_w')
        self.assign_steep = tf.placeholder(
            tf.float32, [batch_size, 1, 1], name='assign_steep')
        self.assign_clipmin = tf.placeholder(
            tf.float32, input_shape, name='assign_clipmin')
        self.assign_clipmax = tf.placeholder(
            tf.float32, input_shape, name='assign_clipmax')

        # Change of variables
        self.new_q = (tf.tanh(modifier + q_var) + 1) / 2
        self.new_q = self.new_q * (clipmax_var - clipmin_var) + clipmin_var
        # Distance to the input data
        orig = (tf.tanh(q_var) + 1) / \
            2 * (clipmax_var - clipmin_var) + clipmin_var
        # L2 perturbation loss
        l2dist = tf.reduce_sum(tf.square(self.new_q - orig), input_axis)
        self.l2dist = tf.maximum(0., l2dist - eps**2)

        def sigmoid(x, a=1):
            return 1 / (1 + tf.exp(-a * x))

        self.new_q_rs = tf.reshape(self.new_q, (batch_size, 1, -1))
        self.new_q_rs = self.new_q_rs / \
            tf.norm(self.new_q_rs, axis=2, keepdims=True)
        x_var_rs = tf.reshape(x_var, (batch_size, m, -1))
        dist = tf.norm(self.new_q_rs - x_var_rs, axis=2, keepdims=True)
        self.nn_loss = tf.reduce_sum(
            w_var * sigmoid(min_dist - dist, steep_var), (1, 2))

        # Setup optimizer
        if self.ord == 2:
            # For L-2 norm constraint, we use a penalty term
            self.loss = tf.reduce_mean(const_var * self.nn_loss + self.l2dist)
        elif self.ord == np.inf:
            self.loss = tf.reduce_mean(self.nn_loss)
        else:
            raise ValueError('Invalid choice of perturbation norm!')

        start_vars = set(x.name for x in tf.global_variables())
        optimizer = tf.train.AdamOptimizer(self.learning_rate)
        self.train_step = optimizer.minimize(self.loss, var_list=[modifier])
        end_vars = tf.global_variables()
        new_vars = [x for x in end_vars if x.name not in start_vars]

        self.setup = []
        self.setup.append(q_var.assign(self.assign_q))
        self.setup.append(const_var.assign(self.assign_const))
        self.setup.append(x_var.assign(self.assign_x))
        self.setup.append(w_var.assign(self.assign_w))
        self.setup.append(steep_var.assign(self.assign_steep))
        self.setup.append(clipmin_var.assign(self.assign_clipmin))
        self.setup.append(clipmax_var.assign(self.assign_clipmax))
        self.init = tf.variables_initializer(var_list=[modifier] + new_vars)

    def attack_batch(self, Q, y_Q, eps):
        bin_search_steps = self.bin_search_steps
        max_iter = self.max_iter
        rnd_start = self.rnd_start

        print("  Finding nn as target...")
        norm = np.sqrt(np.sum(Q**2, axis=1, keepdims=True))
        ind = np.squeeze(norm) != 0
        Q_nm = np.zeros_like(Q)
        Q_nm[ind] = Q[ind] / norm[ind]
        nn = find_2nd_nn_l2(Q_nm, y_Q, self.trnX, self.trny, self.m)
        target = self.trnX[nn]

        # Get weights w
        w = 2 * (self.trny[nn] == y_Q[:, np.newaxis]).astype(np.float32) - 1

        # Initialize steep
        steep = np.ones((self.batch_size, 1, 1)) * 4

        o_bestl2 = np.zeros((self.batch_size, )) + 1e9
        o_bestadv = np.zeros_like(Q[:self.batch_size], dtype=np.float32)

        # Set the lower and upper bounds
        lower_bound = np.zeros(self.batch_size)
        const = np.ones(self.batch_size) * self.init_const
        upper_bound = np.ones(self.batch_size) * 1e9

        if self.ord == np.inf:
            bin_search_steps = 1

        for outer_step in range(bin_search_steps):

            noise = rnd_start * np.random.rand(*Q.shape)
            Q_tanh = np.clip(Q + noise, 0., 1.)

            # Calculate bound with L-inf norm constraints
            if self.ord == np.inf:
                # Re-scale instances to be within range [x-d, x+d]
                # for d is pert_bound
                clipmin = np.clip(Q_tanh - eps, 0., 1.)
                clipmax = np.clip(Q_tanh + eps, 0., 1.)

            # Calculate bound with L2 norm constraints
            elif self.ord == 2:
                # Re-scale instances to be within range [0, 1]
                clipmin = np.zeros_like(Q_tanh)
                clipmax = np.ones_like(Q_tanh)

            Q_tanh = (Q_tanh - clipmin) / (clipmax - clipmin)
            Q_tanh = (Q_tanh * 2) - 1
            Q_tanh = np.arctanh(Q_tanh * .999999)
            Q_batch = Q_tanh[:self.batch_size]

            bestl2 = np.zeros((self.batch_size, )) + 1e9
            bestadv = np.zeros_like(Q_batch, dtype=np.float32)
            print("  Binary search step {} of {}".format(
                outer_step, bin_search_steps))

            # Set the variables so that we don't have to send them over again
            self.sess.run(self.init)
            setup_dict = {self.assign_q: Q_batch,
                          self.assign_w: w[:, :, np.newaxis],
                          self.assign_x: target,
                          self.assign_const: const,
                          self.assign_steep: steep,
                          self.assign_clipmin: clipmin,
                          self.assign_clipmax: clipmax}
            self.sess.run(self.setup, feed_dict=setup_dict)

            prev = 1e6
            for iteration in range(max_iter):
                # Take one step in optimization
                _, l, l2s, qs, qs_nm = self.sess.run([self.train_step,
                                                      self.loss,
                                                      self.l2dist,
                                                      self.new_q,
                                                      self.new_q_rs])

                if iteration % (max_iter // 10) == 0:
                    print(("    Iteration {} of {}: loss={:.3g} l2={:.3g}").format(
                        iteration, max_iter, l, np.mean(l2s)))

                # Abort early if stop improving
                if self.abort_early and iteration % (max_iter // 10) == 0:
                    if l > prev * .9999:
                        print("    Failed to make progress; stop early")
                        break
                    prev = l

                # Check success of adversarial examples
                check_iter = [int(max_iter * 0.8),
                              int(max_iter * 0.9), int(max_iter - 1)]
                if iteration in check_iter:
                    nn = find_nn(np.squeeze(qs_nm), self.trnX, self.m)
                    y_knn = classify(nn, self.trny)
                    suc_ind = np.where(y_knn != y_Q)[0]
                    print(1 - (len(suc_ind) / self.batch_size))
                    for ind in suc_ind:
                        if l2s[ind] < bestl2[ind]:
                            bestl2[ind] = l2s[ind]
                            bestadv[ind] = qs[ind]

            # Adjust const according to results
            for e in range(self.batch_size):
                if bestl2[e] < 1e9:
                    # Success, divide const by two
                    upper_bound[e] = min(upper_bound[e], const[e])
                    if upper_bound[e] < 1e9:
                        const[e] = (lower_bound[e] + upper_bound[e]) / 2
                    if bestl2[e] < o_bestl2[e]:
                        o_bestl2[e] = bestl2[e]
                        o_bestadv[e] = bestadv[e]
                else:
                    # Failure, either multiply by 10 if no solution found yet
                    #          or do binary search with the known upper bound
                    lower_bound[e] = max(lower_bound[e], const[e])
                    if upper_bound[e] < 1e9:
                        const[e] = (lower_bound[e] + upper_bound[e]) / 2
                    else:
                        const[e] *= 10

        # Also save unsuccessful samples
        # ind = np.where(o_bestl2 == 1e9)[0]
        # o_bestadv[ind] = qs[ind]

        return o_bestadv



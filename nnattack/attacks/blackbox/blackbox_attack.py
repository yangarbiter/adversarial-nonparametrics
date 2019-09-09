"""
https://github.com/LeMinhThong/blackbox-attack/blob/master/blackbox_attack.py
"""
import json
import tempfile
from typing import List
import time
import random
import os

import tensorflow as tf
import keras
from keras.models import model_from_json
import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm
from mkdir_p import mkdir_p

from ..base import AttackModel
from .attackbox import OPT_attack_lf
from ...models.faiss_model import FaissLSHModel


def attack_untargeted(predict_fn, train_dataset, x0, y0, ord, alpha=0.2,
        beta=0.001, iterations=1000, ori_shape=None):
    """ Attack the original image and return adversarial example
        model: (pytorch model)
        train_dataset: set of training data
        (x0, y0): original image
    """
    if isinstance(predict_fn, tuple):
        if predict_fn[0] == 'keras':
            with tf.device('/CPU:0'):
                with open(predict_fn[1] + ".json") as fp:
                    model = model_from_json(json.dumps(json.load(fp)))
                model.load_weights(predict_fn[1])
            def predict_fn(x):
                with tf.device('/CPU:0'):
                    ret = model.predict(np.reshape(x, [len(x)] + list(ori_shape[1:]))).argmax(1)
                return ret
        elif predict_fn[0] == 'faiss':
            file_path = predict_fn[1]
            model = predict_fn[2]
            model.load(file_path)
            def predict_fn(x):
                return model.predict(np.reshape(x, [len(x), -1]))

    #if (model.predict([x0]) != y0):
    if (predict_fn([x0]) != y0):
        #print("Fail to classify the image. No need to attack.")
        return x0

    num_samples = min(1000, len(train_dataset))
    best_theta, g_theta = None, float('inf')
    query_count = 0

    #print("Searching for the initial direction on %d samples: " % (num_samples))
    #timestart = time.time()
    samples = set(random.sample(range(len(train_dataset)), num_samples))
    for i, (xi, yi) in enumerate(train_dataset):
        if i not in samples:
            continue
        query_count += 1
        if predict_fn([xi]) != y0:
            theta = xi - x0
            initial_lbd = np.linalg.norm(theta, ord=ord)
            theta = theta / np.linalg.norm(theta, ord=ord)
            lbd, count = fine_grained_binary_search(predict_fn, x0, y0, theta, initial_lbd, g_theta)
            query_count += count
            if lbd < g_theta:
                best_theta, g_theta = theta, lbd
                #print("--------> Found distortion %.4f" % g_theta)

    #timeend = time.time()
    #print("==========> Found best distortion %.4f in %.4f seconds using %d queries" % (g_theta, timeend-timestart, query_count))

    timestart = time.time()
    g1 = 1.0
    theta, g2 = np.copy(best_theta), g_theta
    #torch.manual_seed(0)
    opt_count = 0
    stopping = 0.01
    prev_obj = 100000
    for i in range(iterations):
        #gradient = torch.zeros(theta.size())
        gradient = np.zeros_like(theta)
        q = 10
        min_g1 = float('inf')
        for _ in range(q):
            #u = torch.randn(theta.size()).type(torch.FloatTensor)
            u = np.random.random(theta.shape)
            u = u / np.linalg.norm(u, ord=ord)
            ttt = theta+beta * u
            ttt = ttt/np.linalg.norm(ttt, ord=ord)
            g1, count = fine_grained_binary_search_local(predict_fn, x0, y0, ttt, initial_lbd = g2, tol=beta/500)
            opt_count += count
            gradient += (g1-g2)/beta * u
            if g1 < min_g1:
                min_g1 = g1
                min_ttt = ttt
        gradient = 1.0/q * gradient

        if (i+1)%50 == 0:
            #print("Iteration %3d: g(theta + beta*u) = %.4f g(theta) = %.4f distortion %.4f num_queries %d" % (i+1, g1, g2, np.linalg.norm(g2*theta, ord=ord), opt_count))
            if g2 > prev_obj-stopping:
                break
            prev_obj = g2

        min_theta = theta
        min_g2 = g2

        for _ in range(15):
            new_theta = theta - alpha * gradient
            norm = np.linalg.norm(new_theta, ord=ord)
            if np.isclose(norm, 0) or np.isclose(norm, np.inf):
                break
            new_theta = new_theta / norm
            new_g2, count = fine_grained_binary_search_local(predict_fn, x0, y0, new_theta, initial_lbd = min_g2, tol=beta/500)
            opt_count += count
            alpha = alpha * 2
            if new_g2 < min_g2:
                min_theta = new_theta
                min_g2 = new_g2
            else:
                break

        if min_g2 >= g2:
            for _ in range(15):
                alpha = alpha * 0.25
                new_theta = theta - alpha * gradient
                norm = np.linalg.norm(new_theta, ord=ord)
                if np.isclose(norm, 0) or np.isclose(norm, np.inf):
                    break
                new_theta = new_theta/norm
                new_g2, count = fine_grained_binary_search_local(predict_fn, x0, y0, new_theta, initial_lbd = min_g2, tol=beta/500)
                opt_count += count
                if new_g2 < g2:
                    min_theta = new_theta
                    min_g2 = new_g2
                    break

        if min_g2 <= min_g1:
            theta, g2 = min_theta, min_g2
        else:
            theta, g2 = min_ttt, min_g1

        if g2 < g_theta:
            best_theta, g_theta = np.copy(theta), g2

        #print(alpha)
        if alpha < 1e-4:
            alpha = 1.0
            #print("Warning: not moving, g2 %lf gtheta %lf" % (g2, g_theta))
            beta = beta * 0.1
            if (beta < 0.0005):
                break

    target = predict_fn([x0 + g_theta*best_theta])
    timeend = time.time()
    print("\nAdversarial Example Found Successfully: distortion %.4f target %d queries %d \nTime: %.4f seconds" % (g_theta, target, query_count + opt_count, timeend-timestart))
    return x0 + g_theta*best_theta

def fine_grained_binary_search_local(predict_fn, x0, y0, theta, initial_lbd = 1.0, tol=1e-5):
    nquery = 0
    lbd = initial_lbd

    if predict_fn([x0+lbd*theta]) == y0:
        lbd_lo = lbd
        lbd_hi = lbd*1.01
        nquery += 1
        while predict_fn([x0+lbd_hi*theta]) == y0:
            lbd_hi = lbd_hi*1.01
            nquery += 1
            if lbd_hi > 20:
                return float('inf'), nquery
    else:
        lbd_hi = lbd
        lbd_lo = lbd*0.99
        nquery += 1
        while predict_fn([x0+lbd_lo*theta]) != y0 :
            lbd_lo = lbd_lo*0.99
            nquery += 1

    while (lbd_hi - lbd_lo) > tol:
        lbd_mid = (lbd_lo + lbd_hi)/2.0
        nquery += 1
        if predict_fn([x0 + lbd_mid*theta]) != y0:
            lbd_hi = lbd_mid
        else:
            lbd_lo = lbd_mid
    return lbd_hi, nquery

def fine_grained_binary_search(predict_fn, x0, y0, theta, initial_lbd, current_best):
    nquery = 0
    if initial_lbd > current_best:
        if predict_fn([x0+current_best*theta]) == y0:
            nquery += 1
            return float('inf'), nquery
        lbd = current_best
    else:
        lbd = initial_lbd

    lbd_hi = lbd
    lbd_lo = 0.0

    while (lbd_hi - lbd_lo) > 1e-5:
        lbd_mid = (lbd_lo + lbd_hi)/2.0
        nquery += 1
        if predict_fn([x0 + lbd_mid*theta]) != y0:
            lbd_hi = lbd_mid
        else:
            lbd_lo = lbd_mid
    return lbd_hi, nquery



class BlackBoxAttack(AttackModel):

    def __init__(self, ord, model, random_state=None):
        super().__init__(ord=ord)
        self.model = model
        self.random_state = random_state
        self.alpha = 2
        self.beta = 0.05

    def perturb(self, X: np.ndarray, y, eps):
        ori_shape = X.shape
        X = X.reshape((len(X), -1)).astype(np.float32)
        ret: List[np.ndarray] = []
        dataset = list(zip(X, y))

        if hasattr(self.model, 'model') and isinstance(self.model.model, keras.models.Model):
            filen = tempfile.mkdtemp(prefix="/tmp2/aptemp/")
            filen = os.path.join(filen, "model")

            with open(filen + ".json", "w") as fp:
                fp.write(self.model.model.to_json())
            self.model.model.save_weights(filen)
            predict_fn = ('keras', filen)
        elif 'faiss' in self.model.__module__:
            _, filen = tempfile.mkstemp(prefix="/tmp2/aptemp/")
            self.model.save(filen)
            temp_model = eval(self.model.__repr__())
            temp_model.trnX = self.model.trnX
            temp_model.trny = self.model.trny
            predict_fn = ('faiss', filen, temp_model)

        else:
            def predict_fn(x):
                return self.model.predict(np.reshape(x, [len(x)] + list(ori_shape[1:])))

        if self.ord == np.inf:
            attacker = OPT_attack_lf(self.model)
            ret = Parallel(n_jobs=1, verbose=5)(delayed(attacker)(
                xi, yi, TARGETED=False) for (xi, yi) in dataset)
        else:
            ret = Parallel(n_jobs=10, verbose=10)(delayed(attack_untargeted)(
                predict_fn, dataset, xi, yi, self.ord, alpha=self.alpha, beta=self.beta,
                iterations=1000, ori_shape=ori_shape) for (xi, yi) in dataset)

        ret = (np.vstack(ret) - X).astype(np.float32)
        self.perts = ret
        if isinstance(eps, list):
            rret = []
            norms = np.linalg.norm(ret, axis=1, ord=self.ord)
            for ep in eps:
                t = np.copy(ret)
                t[norms > ep, :] = 0
                rret.append(t.reshape(ori_shape))
            return rret
        elif eps is not None:
            ret[np.linalg.norm(ret, axis=1, ord=self.ord) > eps, :] = 0
            return ret.reshape(ori_shape)
        else:
            return ret.reshape(ori_shape)

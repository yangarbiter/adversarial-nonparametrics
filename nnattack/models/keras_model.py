import itertools
import threading

from cleverhans.attacks import ProjectedGradientDescent, FastGradientMethod
from cleverhans.utils_keras import KerasModelWrapper
from cleverhans.loss import CrossEntropy
from cleverhans.train import train
from cleverhans.utils_tf import initialize_uninitialized_global_variables

import tensorflow as tf

import keras
from keras.models import Model, clone_model
from keras.layers import Dense, Input
from keras.optimizers import Adam, Nadam
from keras.regularizers import l2

import numpy as np
from sklearn.base import BaseEstimator

from .robust_nn.eps_separation import find_eps_separated_set

def get_adversarial_acc_metric(model, fgsm, fgsm_params):
    def adv_acc(y, _):
        # Generate adversarial examples
        x_adv = fgsm.generate(model.get_input_at(0), **fgsm_params)
        # Consider the attack to be constant
        x_adv = tf.stop_gradient(x_adv)

        # Accuracy on the adversarial examples
        preds_adv = model(x_adv)
        return keras.metrics.categorical_accuracy(y, preds_adv)
    return adv_acc

def get_adversarial_loss(model, fgsm, fgsm_params):
    def adv_loss(y, preds):
        # Cross-entropy on the legitimate examples
        cross_ent = keras.losses.categorical_crossentropy(y, preds)

        # Generate adversarial examples
        x_adv = fgsm.generate(model.get_input_at(0), **fgsm_params)
        # Consider the attack to be constant
        x_adv = tf.stop_gradient(x_adv)

        # Cross-entropy on the adversarial examples
        preds_adv = model(x_adv)
        cross_ent_adv = keras.losses.categorical_crossentropy(y, preds_adv)

        return 0.5 * cross_ent + 0.5 * cross_ent_adv
    return adv_loss

def logistic_regression(input_x, input_shape, n_classes, l2_weight=0.0, **kwargs):
    inputs = Input(shape=input_shape, tensor=input_x)
    x = Dense(n_classes, activation='softmax', kernel_regularizer=l2(l2_weight))(inputs)

    return Model(inputs=[inputs], outputs=[x]), None

def mlp(input_x, input_shape, n_classes, l2_weight=0.0, **kwargs):
    inputs = Input(shape=input_shape, tensor=input_x)
    x = Dense(128, activation='relu', kernel_regularizer=l2(l2_weight))(inputs)
    x = Dense(n_classes, activation='softmax', kernel_regularizer=l2(l2_weight))(x)

    return Model(inputs=[inputs], outputs=[x]), None

class KerasModel(BaseEstimator):
    def __init__(self, lbl_enc, n_features, n_classes, sess,
            learning_rate=1e-3, batch_size=128, epochs=20, optimizer='adam',
            l2_weight=1e-5, architecture='arch_001', random_state=None,
            attacker=None, callbacks=None, train_type:str=None, eps:float=0.1,
            ord=np.inf, eps_list=None):
        keras.backend.set_session(sess)
        self.n_features = n_features
        self.n_classes = n_classes
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.architecture = architecture
        self.epochs = epochs
        self.lbl_enc = lbl_enc
        self.optimizer_name = optimizer
        if optimizer == 'nadam':
            self.optimizer = Nadam()
        elif optimizer == 'adam':
            self.optimizer = Adam(lr=self.learning_rate)
        self.l2_weight = l2_weight
        self.callbacks=callbacks
        self.loss = 'categorical_crossentropy'
        self.random_state = random_state
        self.train_type = train_type

        input_shape = tuple(n_features)
        model, self.preprocess_fn = globals()[self.architecture](
            None, input_shape, n_classes, self.l2_weight)
        #model.summary()
        self.model = model

        ### Attack ####
        if eps_list is None:
            eps_list = [e*0.01 for e in range(100)]
        else:
            eps_list = [e for e in eps_list]
        self.sess = sess
        self.eps = eps
        self.ord = ord
        ###############

    def fit(self, X, y, sample_weight=None):
        if self.train_type is not None:
            pass

        if self.train_type == 'adv':
            Y = self.lbl_enc.transform(y.reshape(-1, 1))
            wrap_2 = KerasModelWrapper(self.model)
            fgsm_2 = ProjectedGradientDescent(wrap_2, sess=self.sess)
            self.model(self.model.input)
            fgsm_params = {'eps': self.eps}

            # Use a loss function based on legitimate and adversarial examples
            adv_loss_2 = get_adversarial_loss(self.model, fgsm_2, fgsm_params)
            adv_acc_metric_2 = get_adversarial_acc_metric(self.model, fgsm_2, fgsm_params)
            self.model.compile(
                optimizer=keras.optimizers.Nadam(),
                loss=adv_loss_2,
                metrics=['accuracy', adv_acc_metric_2]
            )
            self.model.fit(X, Y,
                batch_size=self.batch_size,
                epochs=self.epochs,
                verbose=2,
                sample_weight=sample_weight,
            )

            self.augX, self.augy = None, None

        elif self.train_type == 'advPruning':
            y = y.astype(int)*2-1
            self.augX, self.augy = find_eps_separated_set(
                    X, self.eps/2, y, ord=self.ord)
            self.augy = (self.augy+1)//2

            self.model.compile(loss=self.loss, optimizer=self.optimizer, metrics=[])
            Y = self.lbl_enc.transform(self.augy.reshape(-1, 1))
            self.model.fit(self.augX, Y, batch_size=self.batch_size, verbose=0,
                        epochs=self.epochs, sample_weight=sample_weight)
            print("number of augX", np.shape(self.augX), len(self.augy))
        elif self.train_type is None:
            self.model.compile(loss=self.loss, optimizer=self.optimizer, metrics=[])
            Y = self.lbl_enc.transform(y.reshape(-1, 1))
            self.model.fit(X, Y, batch_size=self.batch_size, verbose=0,
                        epochs=self.epochs, sample_weight=sample_weight)
        else:
            raise ValueError("Not supported train type: %s", self.train_type)

    def predict(self, X):
        X = np.asarray(X)
        if self.preprocess_fn is not None:
            X = self.preprocess_fn(X)
        pred = self.model.predict(X)
        return pred.argmax(1)
        #return self.lbl_enc.inverse_transform(pred).reshape(-1)

    def predict_proba(self, X):
        X = np.asarray(X)
        if self.preprocess_fn is not None:
            X = self.preprocess_fn(X)
        pred = self.model.predict(X)
        return np.hstack((1-pred, pred))

    def score(self, X, y):
        pred = self.predict(X)
        return (pred == y).mean()

    def _get_pert(self, X, Y, eps:float, model):
        x = tf.placeholder(tf.float32, shape=([None] + list(self.n_features)))
        y = tf.placeholder(tf.float32, shape=(None, self.n_classes))

        wrap = KerasModelWrapper(model)
        pgd = ProjectedGradientDescent(wrap, ord=self.ord, sess=self.sess)
        if eps >= 0.05:
            adv_x = pgd.generate(x, y=y, eps=eps)
        else:
            adv_x = pgd.generate(x, y=y, eps=eps, eps_iter=eps)
        adv_x = tf.stop_gradient(adv_x)
        ret = adv_x - x
        return ret.eval(feed_dict={x: X, y: Y}, session=self.sess)

    def perturb(self, X, y, eps=0.1):
        if len(y.shape) == 1:
            Y = self.lbl_enc.transform(y.reshape(-1, 1))
        else:
            Y = y
        #Y[:, 0], Y[:, 1] = Y[:, 1], Y[:, 0]
        if isinstance(eps, list):
            rret = []
            for ep in eps:
                rret.append(self._get_pert(X, Y, ep, self.model))
            return rret
        elif isinstance(eps, float):
            ret = self._get_pert(X, Y, eps, self.model)
        else:
            raise ValueError
        return ret


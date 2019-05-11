import itertools
import threading

from cleverhans.attacks import ProjectedGradientDescent, FastGradientMethod
from cleverhans.utils_keras import KerasModelWrapper
from cleverhans.loss import CrossEntropy
from cleverhans.train import train
from cleverhans.utils_tf import initialize_uninitialized_global_variables

import tensorflow as tf
#import tensorflow.keras as keras
#from tensorflow.keras.models import Model
#from tensorflow.keras.layers import Dense, Input
#from tensorflow.keras.optimizers import Adam, Nadam
#from tensorflow.keras.regularizers import l2
#from tensorflow.keras.models import clone_model

import keras
from keras.models import Model, clone_model
from keras.layers import Dense, Input
from keras.optimizers import Adam, Nadam
from keras.regularizers import l2

import numpy as np
from sklearn.base import BaseEstimator
#from sklearn.linear_model import LogisticRegression

from .robust_nn.eps_separation import find_eps_separated_set

def get_adversarial_acc_metric(model, fgsm, fgsm_params):
    def adv_acc(y, _):
        # Generate adversarial examples
        #x_adv = fgsm.generate(model.input, **fgsm_params)
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
        #x_adv = fgsm.generate(model.input, **fgsm_params)
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

        #init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        #self.sess.run(init_op)

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
        #self.x = tf.placeholder(tf.float32, shape=([None] + list(n_features))) 
        #self.y = tf.placeholder(tf.float32, shape=(None, n_classes))
        ###############

        #self.attacker = None
        #if self.train_type == 'adv':
        #    #self.attacker = attacker(model=self.model)
        #    self.attacker = self

    def fit(self, X, y, sample_weight=None):
        if self.train_type is not None:
            pass

        if self.train_type == 'adv':
            #self.model.compile(loss=self.loss, optimizer=self.optimizer, metrics=[])
            #Y = self.lbl_enc.transform(y.reshape(-1, 1))
            #initialize_uninitialized_global_variables(self.sess)
            #input_generator = InputGenerator(X, Y, sample_weight,
            #    attacker=self.attacker, shuffle=True, batch_size=self.batch_size,
            #    random_state=self.random_state)
            #self.model.fit_generator(
            #    input_generator,
            #    steps_per_epoch=((X.shape[0]*2 - 1) // self.batch_size) + 1,
            #    epochs=self.epochs,
            #    verbose=1,
            #)
            #######################################
            #Y = self.lbl_enc.transform(y.reshape(-1, 1))
            #train_params = {
            #    'init_all': True,
            #    'rng': self.random_state,
            #    'nb_epochs': self.epochs,
            #    'batch_size': self.batch_size,
            #    'learning_rate': self.learning_rate,
            #    'optimizor': tf.train.RMSPropOptimizer,
            #}
            #wrap = KerasModelWrapper(self.model)
            #pgd = ProjectedGradientDescent(wrap, sess=self.sess, nb_iter=20)
            #pgd_params = {'eps': self.eps}
            ##attack = pgd.generate(x, y=y, **pgd_params)
            #def attack(x):
            #    return pgd.generate(x, **pgd_params)
            #loss = CrossEntropy(wrap, smoothing=0.1, attack=attack)
            #def evaluate():
            #    #print("XDDD %f", self.sess.run(loss))
            #    print('Test accuracy on legitimate examples: %0.4f' % self.score(X, y))
            #train(self.sess, loss, X.astype(np.float32), Y.astype(np.float32),
            #        args=train_params, evaluate=evaluate)
            ######################################
            Y = self.lbl_enc.transform(y.reshape(-1, 1))
            wrap_2 = KerasModelWrapper(self.model)
            fgsm_2 = ProjectedGradientDescent(wrap_2, sess=self.sess)
            self.model(self.model.input)
            fgsm_params = {'eps': self.eps}

            # Use a loss function based on legitimate and adversarial examples
            adv_loss_2 = get_adversarial_loss(self.model, fgsm_2, fgsm_params)
            adv_acc_metric_2 = get_adversarial_acc_metric(self.model, fgsm_2, fgsm_params)
            self.model.compile(
                #optimizer=keras.optimizers.Adam(self.learning_rate),
                optimizer=keras.optimizers.Nadam(),
                loss=adv_loss_2,
                metrics=['accuracy', adv_acc_metric_2]
            )
            self.model.fit(X, Y,
                batch_size=self.batch_size,
                epochs=self.epochs,
                verbose=2
            )
            print(self.model.predict(X))
            print((self.model.predict(X).argmax(1) == y).mean())

            #pre_model = clone_model(self.model)
            #pre_model.compile(loss=self.loss, optimizer=self.optimizer, metrics=[])
            #Y = self.lbl_enc.transform(y.reshape(-1, 1))
            #pre_model.fit(X, Y, batch_size=self.batch_size, verbose=1,
            #            epochs=self.epochs, sample_weight=sample_weight)

            #pert = self._get_pert(X, Y, eps=self.eps, model=pre_model)

            #print(tf.global_variables())
            #print(((pre_model.predict(X).argmax(1))==y).mean())
            #print((pre_model.predict(X + pert).argmax(1)==y).mean())

            ##self.augX = np.vstack((X, X+self.perturb(X, y, eps=self.eps)))
            #self.optimizer = Adam(lr=self.learning_rate)
            #self.model.compile(loss=self.loss, optimizer=self.optimizer, metrics=[])
            #self.augX = np.vstack((X, X+pert))
            #self.augy = np.concatenate((y, y))
            ##self.augX = X + pert
            ##self.augy = y
            #Y = self.lbl_enc.transform(self.augy.reshape(-1, 1))

            #self.model.fit(self.augX, Y, batch_size=self.batch_size, verbose=1,
            #            epochs=self.epochs, sample_weight=sample_weight,
            #            shuffle=True)
            #print(((pre_model.predict(X).argmax(1))==y).mean())
            #print((pre_model.predict(X + pert).argmax(1)==y).mean())
            #print(((self.predict(X))==y).mean())
            #print((self.predict(X + pert)==y).mean())
            ##print((self.predict(self.augX + self._get_pert(self.augX, Y, eps=self.eps, model=self.model))==self.augy).mean())
            #clf = LogisticRegression()
            #clf.fit(self.augX, self.augy)

            #Y = self.lbl_enc.transform(y.reshape(-1, 1))
            #print('XD', (clf.predict(X + self._get_pert(X, Y, eps=self.eps, model=self.model))==y).mean())
            #print('XD', (self.predict(X + self._get_pert(X, Y, eps=self.eps, model=self.model))==y).mean())
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

class InputGenerator(object):
    def __init__(self, X, Y=None, sample_weight=None, attacker=None,
                shuffle=False, batch_size=256, eps:float=0.1, random_state=None):
        self.X = X
        self.Y = Y
        self.lock = threading.Lock()
        if random_state is None:
            random_state = np.random.RandomState()
        if attacker is not None:
            # assume its a multiple of 2
            batch_size = batch_size // 2
        self.index_generator = self._flow_index(X.shape[0], batch_size, shuffle,
                random_state)
        self.attacker = attacker
        self.sample_weight = sample_weight
        self.eps = eps

    def __iter__(self):
        return self

    def __next__(self, *args, **kwargs):
        return self.next(*args, **kwargs)

    def _flow_index(self, n, batch_size, shuffle, random_state):
        index = np.arange(n)
        for epoch_i in itertools.count():
            if shuffle:
                random_state.shuffle(index)
            for batch_start in range(0, n, batch_size):
                batch_end = min(batch_start + batch_size, n)
                yield epoch_i, index[batch_start: batch_end]

    def next(self):
        with self.lock:
            _, index_array = next(self.index_generator)
        batch_X = self.X[index_array]

        if self.Y is None:
            return batch_X
        else:
            batch_Y = self.Y[index_array]
            if self.attacker is not None:
                adv_X = batch_X + self.attacker.perturb(batch_X, batch_Y, eps=self.eps)
                batch_X = np.concatenate((batch_X, adv_X), axis=0)

            if self.sample_weight is not None:
                batch_weight = self.sample_weight[index_array]
                if self.attacker is not None:
                    batch_Y = np.concatenate((batch_Y, batch_Y), axis=0)
                    batch_weight = np.concatenate((batch_weight, batch_weight), axis=0)
                return batch_X, batch_Y, batch_weight
            else:
                if self.attacker is not None:
                    batch_Y = np.concatenate((batch_Y, batch_Y), axis=0)
                return batch_X, batch_Y

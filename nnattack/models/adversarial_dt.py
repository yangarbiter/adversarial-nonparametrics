import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from .robust_nn.eps_separation import find_eps_separated_set

class AdversarialDt(DecisionTreeClassifier):
    def __init__(self, **kwargs):
        self.ord = kwargs.pop("ord", np.inf)
        self.eps = kwargs.pop("eps", np.inf)
        self.attack_model = kwargs.pop("attack_model", None)
        self.train_type = kwargs.pop("train_type", 'adv')
        super().__init__(**kwargs)

    def fit(self, X, y, eps:float=None):
        print("original X", np.shape(X), len(y))
        if self.train_type == 'adv':
            if eps is None and self.eps is None:
                raise ValueError("eps should not be None with train type %s", self.train_type)
            elif eps is None:
                eps = self.eps

            advX = self.attack_model.perturb(X, y=y, eps=eps)

            ind = np.where(np.linalg.norm(advX, axis=1) != 0)

            self.augX = np.vstack((X, X[ind]+advX[ind]))
            self.augy = np.concatenate((y, y[ind]))
        elif self.train_type == 'robustv1':
            if eps is None and self.eps is None:
                raise ValueError("eps should not be None with train type %s", self.train_type)
            elif eps is None:
                eps = self.eps

            y = y.astype(int)*2-1
            self.augX, self.augy = find_eps_separated_set(X, eps/2, y, self.ord)
            self.augy = (self.augy+1)//2
        elif self.train_type is None:
            self.augX, self.augy = X, y
        else:
            raise ValueError("Not supported training type %s", self.train_type)

        print("number of augX", np.shape(self.augX), len(self.augy))

        return super().fit(self.augX, self.augy)

class AdversarialRf(RandomForestClassifier):
    def __init__(self, **kwargs):
        self.ord = kwargs.pop("ord", np.inf)
        self.eps = kwargs.pop("eps", np.inf)
        self.attack_model = kwargs.pop("attack_model", None)
        self.train_type = kwargs.pop("train_type", None)
        super().__init__(**kwargs)

    def fit(self, X, y, eps:float=None):
        print("original X", np.shape(X), len(y))
        if self.train_type == 'adv':
            if eps is None and self.eps is None:
                raise ValueError("eps should not be None with train type %s", self.train_type)
            elif eps is None:
                eps = self.eps

            advX = self.attack_model.perturb(X, y=y, eps=eps)

            ind = np.where(np.linalg.norm(advX, axis=1) != 0)

            self.augX = np.vstack((X, X[ind]+advX[ind]))
            self.augy = np.concatenate((y, y[ind]))
        elif self.train_type == 'robustv1':
            if eps is None and self.eps is None:
                raise ValueError("eps should not be None with train type %s", self.train_type)
            elif eps is None:
                eps = self.eps

            y = y.astype(int)*2-1
            self.augX, self.augy = find_eps_separated_set(X, eps/2, y, self.ord)
            self.augy = (self.augy+1)//2
        elif self.train_type is None:
            self.augX, self.augy = X, y
        else:
            raise ValueError("Not supported training type %s", self.train_type)

        print("number of augX", np.shape(self.augX), len(self.augy))

        return super().fit(self.augX, self.augy)

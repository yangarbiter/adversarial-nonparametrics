import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from .robust_nn.eps_separation import find_eps_separated_set
from .robust_nn.robust_1nn import get_aug_v2

def get_aug_data(model, X, y, eps):
    if model.train_type in ['adv', 'robustv1', 'robustv2']:
        if eps is None and model.eps is None:
            raise ValueError("eps should not be None with train type %s", model.train_type)
        elif eps is None:
            eps = model.eps

    if model.train_type == 'adv':
        advX = model.attack_model.perturb(X, y=y, eps=eps)

        ind = np.where(np.linalg.norm(advX, axis=1) != 0)

        augX = np.vstack((X, X[ind]+advX[ind]))
        augy = np.concatenate((y, y[ind]))

    elif model.train_type == 'robustv1':
        if len(np.unique(y)) != 2:
            raise ValueError("Can only deal with number of classes = 2"
                             "got %d", len(np.unique(y)))
        y = y.astype(int)*2-1
        augX, augy = find_eps_separated_set(X, eps/2, y, model.ord)
        augy = (augy+1)//2

    elif model.train_type == 'robustv2':
        if len(np.unique(y)) != 2:
            raise ValueError("Can only deal with number of classes = 2"
                             "got %d", len(np.unique(y)))
        y = y.astype(int)*2-1

        Delta = model.Delta
        delta = model.delta
        augX, augy = get_aug_v2(X, y, Delta, delta, eps, model.ord)
        augy = (augy+1)//2

    elif model.train_type is None:
        augX, augy = X, y
    else:
        raise ValueError("Not supported training type %s", model.train_type)

    return augX, augy


class AdversarialDt(DecisionTreeClassifier):
    def __init__(self, **kwargs):
        self.ord = kwargs.pop("ord", np.inf)
        self.eps = kwargs.pop("eps", 0.1)
        self.attack_model = kwargs.pop("attack_model", None)
        self.train_type = kwargs.pop("train_type", 'adv')
        self.delta = kwargs.pop("delta", 0.1)
        self.Delta = kwargs.pop("Delta", 0.45)
        super().__init__(**kwargs)

    def fit(self, X, y, eps:float=None):
        print("original X", np.shape(X), len(y))
        self.augX, self.augy = get_aug_data(self, X, y, eps)
        print("number of augX", np.shape(self.augX), len(self.augy))
        return super().fit(self.augX, self.augy)

class AdversarialRf(RandomForestClassifier):
    def __init__(self, **kwargs):
        self.ord = kwargs.pop("ord", np.inf)
        self.eps = kwargs.pop("eps", 0.1)
        self.attack_model = kwargs.pop("attack_model", None)
        self.train_type = kwargs.pop("train_type", None)
        self.delta = kwargs.pop("delta", 0.1)
        self.Delta = kwargs.pop("Delta", 0.45)
        super().__init__(**kwargs)

    def fit(self, X, y, eps:float=None):
        print("original X", np.shape(X), len(y))
        self.augX, self.augy = get_aug_data(self, X, y, eps)
        print("number of augX", np.shape(self.augX), len(self.augy))

        return super().fit(self.augX, self.augy)

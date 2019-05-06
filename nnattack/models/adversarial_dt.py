import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from .defense import get_aug_data

class AdversarialDt(DecisionTreeClassifier):
    def __init__(self, **kwargs):
        print(kwargs)
        self.ord = kwargs.pop("ord", np.inf)
        self.sep_measure = kwargs.pop("sep_measure", None)
        self.attack_model = kwargs.pop("attack_model", None)
        self.train_type = kwargs.pop("train_type", 'adv')
        self.delta = kwargs.pop("delta", 0.1)
        self.Delta = kwargs.pop("Delta", 0.45)
        #self.eps = kwargs.pop("eps", 0.1)

        if self.train_type is None:
            pass
        elif self.train_type == 'robust':
            kwargs['splitter'] = 'robust'
            #kwargs['eps'] = self.eps
        elif self.train_type[:7] == 'robust_':
            # for hybrid
            kwargs['splitter'] = 'robust'
            #kwargs['eps'] = self.eps
            self.train_type = self.train_type[7:]

        # The modified DecisionTreeClassifier eats the esp argument
        super().__init__(**kwargs)

    def fit(self, X, y, eps:float=None):
        print("original X", np.shape(X), len(y))
        self.augX, self.augy = get_aug_data(self, X, y, eps)
        print("number of augX", np.shape(self.augX), len(self.augy))
        return super().fit(self.augX, self.augy)

class AdversarialRf(RandomForestClassifier):
    def __init__(self, **kwargs):
        print(kwargs)
        self.ord = kwargs.pop("ord", np.inf)
        #self.eps = kwargs.pop("eps", 0.1)
        self.sep_measure = kwargs.pop("sep_measure", None)
        self.attack_model = kwargs.pop("attack_model", None)
        self.train_type = kwargs.pop("train_type", None)
        self.delta = kwargs.pop("delta", 0.1)
        self.Delta = kwargs.pop("Delta", 0.45)

        if self.train_type is None:
            pass
        elif self.train_type == 'robust':
            kwargs['splitter'] = 'robust'
            #kwargs['eps'] = self.eps
            #kwargs['n_jobs'] = 4
        elif self.train_type[:7] == 'robust_':
            # for hybrid
            kwargs['splitter'] = 'robust'
            #kwargs['eps'] = self.eps
            self.train_type = self.train_type[7:]

        # The modified RandomForestClassifier eats the esp argument
        super().__init__(**kwargs)

    def fit(self, X, y, eps:float=None):
        print("original X", np.shape(X), len(y))
        self.augX, self.augy = get_aug_data(self, X, y, eps)
        print("number of augX", np.shape(self.augX), len(self.augy))

        return super().fit(self.augX, self.augy)

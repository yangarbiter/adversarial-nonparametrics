import numpy as np
from sklearn.tree import DecisionTreeClassifier

from .robust_nn.eps_separation import find_eps_separated_set

class AdversarialDt(DecisionTreeClassifier):
    def __init__(self, **kwargs):
        print(kwargs)
        self.ord = kwargs.pop("ord", np.inf)
        self.eps = kwargs.pop("eps", 0.1)
        self.attack_model = kwargs.pop("attack_model", None)
        self.train_type = kwargs.pop("train_type", 'adv')
        super().__init__(**kwargs)

    def fit(self, X, y):
        print("original X", np.shape(X), len(y))
        if self.train_type == 'adv':
            advX = self.attack_model.perturb(X, y=y, eps=self.eps)

            ind = np.where(np.linalg.norm(advX, axis=1) != 0)

            self.augX = np.vstack((X, X[ind]+advX[ind]))
            self.augy = np.concatenate((y, y[ind]))
        elif self.train_type == 'robustv1':
            y = y.astype(int)*2-1
            self.augX, self.augy = find_eps_separated_set(X, self.eps/2, y)
            self.augy = (self.augy+1)//2
        else:
            raise ValueError("Not supported training type %s", self.train_type)

        print("number of augX", np.shape(self.augX), len(self.augy))

        return super().fit(self.augX, self.augy)

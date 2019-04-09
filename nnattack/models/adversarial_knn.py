import numpy as np
from sklearn.neighbors import KNeighborsClassifier

from .robust_nn.eps_separation import find_eps_separated_set

class AdversarialKnn(KNeighborsClassifier):
    def __init__(self, n_neighbors=5, weights='uniform', algorithm='auto',
                leaf_size=30, p=2, metric='minkowski', metric_params=None,
                n_jobs=1, eps:float = None, **kwargs):
        print(kwargs)
        self.ord = kwargs.pop("ord", np.inf)
        self.attack_model = kwargs.pop("attack_model", None)
        self.eps = eps

        self.train_type = kwargs.pop("train_type", 'adv')

        super().__init__(n_neighbors=n_neighbors, weights=weights,
                    algorithm=algorithm, leaf_size=leaf_size, p=p, metric=metric,
                    metric_params=metric_params, n_jobs=n_jobs, **kwargs)

    def fit(self, X, y, eps:float=None):
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

            if len(np.unique(y)) != 2:
                raise ValueError("Can only deal with number of classes = 2"
                                 "got %d", len(np.unique(y)))

            y = y.astype(int)*2-1
            self.augX, self.augy = find_eps_separated_set(X, eps/2, y, self.ord)
            self.augy = (self.augy+1)//2
        elif self.train_type is None:
            self.augX, self.augy = X, y
        else:
            raise ValueError("Not supported training type %s", self.train_type)

        print("number of augX", np.shape(self.augX), len(self.augy))


        return super().fit(self.augX, self.augy)

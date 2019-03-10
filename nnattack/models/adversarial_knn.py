import numpy as np
from sklearn.neighbors import KNeighborsClassifier

from .robust_nn.eps_separation import find_eps_separated_set

class AdversarialKnn(KNeighborsClassifier):
    def __init__(self, n_neighbors=5, weights='uniform', algorithm='auto',
                leaf_size=30, p=2, metric='minkowski', metric_params=None, n_jobs=1,
                **kwargs):
        print(kwargs)
        self.ord = kwargs.pop("ord", np.inf)
        self.eps = kwargs.pop("eps", 0.1)
        self.attack_model = kwargs.pop("attack_model", None)

        self.train_type = kwargs.pop("train_type", 'adv')
        self.r = kwargs.pop("r", 0.1)

        super().__init__(n_neighbors=n_neighbors, weights=weights,
                    algorithm=algorithm, leaf_size=leaf_size, p=p, metric=metric,
                    metric_params=metric_params, n_jobs=n_jobs, **kwargs)

    def fit(self, X, y):
        if self.train_type == 'adv':
            advX = self.attack_model.perturb(X, y=y, eps=self.eps)

            ind = np.where(np.linalg.norm(advX, axis=1) != 0)

            self.augX = np.vstack((X, X[ind]+advX[ind]))
            self.augy = np.concatenate((y, y[ind]))
        elif self.train_type == 'robustv1':
            y = y.astype(int)*2-1
            self.augX, self.augy = find_eps_separated_set(X, self.r/2, y)
            self.augy = (self.augy+1)//2
        else:
            raise ValueError("Not supported training type %s", self.train_type)

        print("number of augX", np.shape(self.augX), len(self.augy))


        return super().fit(self.augX, self.augy)

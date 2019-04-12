import numpy as np
from sklearn.neighbors import KNeighborsClassifier

from .robust_nn.eps_separation import find_eps_separated_set
from .adversarial_dt import get_aug_data

class AdversarialKnn(KNeighborsClassifier):
    def __init__(self, n_neighbors=5, weights='uniform', algorithm='auto',
                leaf_size=30, p=2, metric='minkowski', metric_params=None,
                n_jobs=1, eps:float = None, **kwargs):
        print(kwargs)
        self.ord = kwargs.pop("ord", np.inf)
        self.attack_model = kwargs.pop("attack_model", None)
        self.eps = eps

        self.train_type = kwargs.pop("train_type", 'adv')
        self.delta = kwargs.pop("delta", 0.1)
        self.Delta = kwargs.pop("Delta", 0.45)

        super().__init__(n_neighbors=n_neighbors, weights=weights,
                    algorithm=algorithm, leaf_size=leaf_size, p=p, metric=metric,
                    metric_params=metric_params, n_jobs=n_jobs, **kwargs)

    def fit(self, X, y, eps:float=None):
        self.augX, self.augy = get_aug_data(self, X, y, eps)
        print("number of augX", np.shape(self.augX), len(self.augy))


        return super().fit(self.augX, self.augy)

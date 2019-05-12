import numpy as np
from sklearn.neighbors import KNeighborsClassifier

from .defense import get_aug_data

class AdversarialKnn(KNeighborsClassifier):
    def __init__(self, sep_measure=None, ord=np.inf, attack_model=None,
                train_type:str=None, delta=0.1, Delta=0.45,
                n_neighbors=5, weights='uniform', algorithm='auto',
                leaf_size=30, p=2, metric='minkowski', metric_params=None,
                n_jobs=1, eps:float = None):
        """K Nearest Neighbors Classifier with defense

        Keyword Arguments:
            ord {float} -- adversarial example perturbation measure (default: {np.inf})
            sep_measure {float} -- The distance measure for data, if None, it will be the same as `ord` (default: {None})
            attack_model {Attack Model} -- The Attack Model, only use when `train_type` is 'adv' (default: {None})
            train_type {str} -- None for undefended classifier, 'robustv2' for adversarial pruning, 'adv' for adversarial training (default: {'adv'})
            delta {float} -- parameter for Wang's 1-NN defense (default: {0.1})
            Delta {float} -- parameter for Wang's 1-NN defense (default: {0.45})
            eps {float} -- defense strength (default: {None})

        Other Arguments follows the original scikit-learn argument (sklearn.neighbors.KNeighborsClassifier).
        """

        self.sep_measure = sep_measure
        self.ord = ord
        self.attack_model = attack_model
        self.eps = eps

        self.train_type = train_type
        self.delta = delta
        self.Delta = Delta

        super().__init__(n_neighbors=n_neighbors, weights=weights,
                    algorithm=algorithm, leaf_size=leaf_size, p=p, metric=metric,
                    metric_params=metric_params, n_jobs=n_jobs)

    def fit(self, X, y, eps:float=None):
        self.augX, self.augy = get_aug_data(self, X, y, eps)
        print("number of augX", np.shape(self.augX), len(self.augy))

        return super().fit(self.augX, self.augy)

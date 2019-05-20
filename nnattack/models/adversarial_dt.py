import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from .defense import get_aug_data

class AdversarialDt(DecisionTreeClassifier):
    def __init__(self, ord=np.inf, sep_measure=None, attack_model=None,
                 train_type='adv', **kwargs):
        """Decision Tree Classifier with defense

        Keyword Arguments:
            ord {float} -- adversarial example perturbation measure (default: {np.inf})
            sep_measure {float} -- The distance measure for data, if None, it will be the same as `ord` (default: {None})
            attack_model {Attack Model} -- The Attack Model, only use when `train_type` is 'adv' (default: {None})
            train_type {str} -- None for undefended classifier, 'robustv2' for adversarial pruning, 'adv' for adversarial training (default: {'adv'})

        Other Arguments follows the original scikit-learn argument (sklearn.tree.DecisionTreeClassifier).
        """

        self.ord = ord
        self.sep_measure = sep_measure
        self.attack_model = attack_model
        self.train_type = train_type

        if self.train_type is None:
            pass
        elif self.train_type == 'robust':
            kwargs['splitter'] = 'robust'
            #kwargs['eps'] = self.eps

        # The modified DecisionTreeClassifier eats the esp argument
        super().__init__(**kwargs)

    def fit(self, X, y, eps:float=None):
        print("original X", np.shape(X), len(y))
        self.augX, self.augy = get_aug_data(self, X, y, eps)
        print("number of augX", np.shape(self.augX), len(self.augy))
        return super().fit(self.augX, self.augy)

class AdversarialRf(RandomForestClassifier):
    def __init__(self, ord=np.inf, sep_measure=None, attack_model=None,
                 train_type='adv', **kwargs):
        """Random Forest Classifier with defense

        Keyword Arguments:
            ord {float} -- adversarial example perturbation measure (default: {np.inf})
            sep_measure {float} -- The distance measure for data, if None, it will be the same as `ord` (default: {None})
            attack_model {Attack Model} -- The Attack Model, only use when `train_type` is 'adv' (default: {None})
            train_type {str} -- None for undefended classifier, 'robustv2' for adversarial pruning, 'adv' for adversarial training (default: {'adv'})

        Other Arguments follows the original scikit-learn argument (sklearn.tree.RandomForestClassifier).
        """

        self.ord = ord
        self.sep_measure = sep_measure
        self.attack_model = attack_model
        self.train_type = train_type

        if self.train_type is None:
            pass
        elif self.train_type == 'robust':
            kwargs['splitter'] = 'robust'

        # The modified RandomForestClassifier eats the esp argument
        super().__init__(**kwargs)

    def fit(self, X, y, eps:float=None):
        print("original X", np.shape(X), len(y))
        self.augX, self.augy = get_aug_data(self, X, y, eps)
        print("number of augX", np.shape(self.augX), len(self.augy))

        return super().fit(self.augX, self.augy)

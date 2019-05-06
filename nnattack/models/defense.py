import numpy as np

from .robust_nn.eps_separation import find_eps_separated_set
from .robust_nn.robust_1nn import get_aug_v2

def get_aug_data(model, X, y, eps):
    if model.train_type in ['adv', 'robustv1', 'robustv1min', 'robustv2']:
        if eps is None and model.eps is None:
            raise ValueError("eps should not be None with train type %s", model.train_type)
        elif eps is None:
            eps = model.eps

    sep_measure = model.sep_measure if model.sep_measure else model.ord

    print("XXXXXXXXX %f", eps)

    if model.train_type == 'adv':
        advX = model.attack_model.perturb(X, y=y, eps=eps)

        ind = np.where(np.linalg.norm(advX, axis=1) != 0)

        augX = np.vstack((X, X[ind]+advX[ind]))
        augy = np.concatenate((y, y[ind]))

    elif model.train_type == 'robustv1min':
        if len(np.unique(y)) != 2:
            raise ValueError("Can only deal with number of classes = 2"
                             "got %d", len(np.unique(y)))
        y = y.astype(int)*2-1
        augX, augy = find_eps_separated_set(X, eps/2, y, 'min_measure')
        augy = (augy+1)//2

    elif model.train_type == 'robustv1':
        if len(np.unique(y)) != 2:
            raise ValueError("Can only deal with number of classes = 2"
                             "got %d", len(np.unique(y)))
        y = y.astype(int)*2-1
        augX, augy = find_eps_separated_set(X, eps/2, y, sep_measure)
        augy = (augy+1)//2

    elif model.train_type == 'robustv2':
        if len(np.unique(y)) != 2:
            raise ValueError("Can only deal with number of classes = 2"
                             "got %d", len(np.unique(y)))
        y = y.astype(int)*2-1

        Delta = model.Delta
        delta = model.delta
        augX, augy = get_aug_v2(X, y, Delta, delta, eps, sep_measure)
        augy = (augy+1)//2

    elif model.train_type is None:
        augX, augy = X, y
    elif model.train_type == 'robust':
        augX, augy = X, y
    else:
        raise ValueError("Not supported training type %s", model.train_type)

    return augX, augy


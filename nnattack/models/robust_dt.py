import numpy as np


def min_info(x, y, n0, n1):
    dn0, dn1 = 0, 0
    min_diff = np.abs(len(n0) / (y == 0).sum() - len(n1) / (y == 1).sum())
    dI 


def split(x, y, eps):
    idx = np.argsort(x)
    x, y = x[idx], y[idx]
    for m in range(len(x)-1):
        eta = (x[m] + x[m+1]) / 2
        dI, IR, IL = [], [], []
        IL, IR = x[x <= eta - eps], x[x >= eta + eps]
        dI = x[np.logical_and((eta-eps) < x, x < (eta+eps))]
        n0 = x[np.logical_and(y == 0, x <= eta - eps)]
        n1 = x[np.logical_and(y == 1, x <= eta - eps)]




class RobustDecisionTree():
    def __init__():

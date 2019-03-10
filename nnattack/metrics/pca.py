import numpy as np
from sklearn.decomposition import PCA as skPCA

from metric_learn.base_metric import BaseMetricLearner

class PCA(object):
    def __init__(self, n_components=None, copy=True, whiten=False,
        svd_solver='auto', tol=0.0, iterated_power='auto',
        random_state=None):
        #super(PCA, self).__init__()
        self.pca_ = skPCA(n_components=n_components, copy=copy, whiten=whiten,
                        svd_solver=svd_solver, tol=tol,
                        iterated_power=iterated_power,
                        random_state=random_state)
        
    def fit(self, X, y):
        self.pca_.fit(X)
        self.pca_.mean_ = None
    
    def transform(self, X):
        return self.pca_.transform(X)

    def transformer(self):
        # components_ : array, shape (n_components, n_features)
        # explained_variance_ : array, shape (n_components,)
        ret = self.pca_.components_
        if self.pca_.whiten:
            ret /= np.sqrt(self.explained_variance_)
        return ret
        #X = check_array(X)
        #if self.mean_ is not None:
        #    X = X - self.mean_
        #X_transformed = np.dot(X, self.components_.T)
        #if self.whiten:
        #    X_transformed /= np.sqrt(self.explained_variance_)
        #return X_transformed
    
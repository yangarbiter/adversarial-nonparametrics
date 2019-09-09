
import numpy as np
import faiss
from sklearn.base import BaseEstimator
from scipy import stats
from .defense import get_aug_data


class FaissLSHModel(BaseEstimator):

    def __init__(self, n_neighbors, n_bits, train_type: str = None,
            eps:float = 0.1, ord=2):
        """
            train_type {str} -- None for undefended classifier, 'advPruning' for adversarial pruning, 'adv' for adversarial training (default: None)
        """
        self.n_neighbors = n_neighbors
        self.n_bits = n_bits
        self.train_type = train_type
        self.eps = eps
        self.ord = ord
        print(self)

    def fit(self, X, y):
        self.augX, self.augy = get_aug_data(self, X, y, self.eps, self.ord)
        augX, augy = self.augX, self.augy

        d = augX.shape[1]
        #self.index = faiss.IndexFlatL2(d)

        self.index = faiss.IndexLSH(d, self.n_bits)

        #quantizer = faiss.IndexFlatL2(d)
        #self.index = faiss.IndexIVFFlat(quantizer, d, 1000)
        #self.index.train(X)

        self.index.add(augX)

        self.trnX, self.trny = augX, augy

    def predict(self, X):
        X = np.asarray(X, np.float32)
        _, I = self.index.search(X, self.n_neighbors)
        query = self.trny[I.reshape(-1)].reshape((len(X), self.n_neighbors))
        return stats.mode(query, axis=1)[0].ravel()

    def save(self, file_path: str):
        faiss.write_index(self.index, file_path)

    def load(self, file_path: str):
        self.index = faiss.read_index(file_path)

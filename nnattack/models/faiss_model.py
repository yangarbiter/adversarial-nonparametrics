
import numpy as np
import faiss
from sklearn.base import BaseEstimator
from scipy import stats


class FaissLSHModel(BaseEstimator):

    def __init__(self, n_neighbors, n_bits):
        self.n_neighbors = n_neighbors
        self.n_bits = n_bits

    def fit(self, X, y):
        d = X.shape[1]
        #self.index = faiss.IndexFlatL2(d)

        self.index = faiss.IndexLSH(d, 200)

        #quantizer = faiss.IndexFlatL2(d)
        #self.index = faiss.IndexIVFFlat(quantizer, d, 1000)
        #self.index.train(X)

        self.index.add(X)

        self.trnX, self.trny = X, y
    
    def predict(self, X):
        X = np.asarray(X, np.float32)
        _, I = self.index.search(X, self.n_neighbors)
        query = self.trny[I.reshape(-1)].reshape((len(X), self.n_neighbors))
        return stats.mode(query, axis=1)[0].ravel()

    def save(self, file_path: str):
        faiss.write_index(self.index, file_path)

    def load(self, file_path: str):
        self.index = faiss.read_index(file_path)

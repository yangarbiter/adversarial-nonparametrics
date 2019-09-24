"""
Approximated adversarial pruning
"""
from __future__ import division
import logging

import faiss
import networkx as nx
from networkx.algorithms.approximation.vertex_cover import min_weighted_vertex_cover
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
from scipy.sparse import csr_matrix

logger = logging.getLogger(__name__)

def approx_ap(X, y, eps, sep_measure, nn_solver="exact", solver_kwargs=None,
        kmeans_clusters: int = 1):
    if isinstance(X, csr_matrix):
        X = X.toarray()

    ori_shape = list(X.shape)
    X = X.reshape((len(X), -1))

    if kmeans_clusters > 1 :
        kmeans = KMeans(n_clusters=kmeans_clusters, n_jobs=10).fit(X)
        kmeans_ind = kmeans.labels_
    else:
        kmeans_ind = np.zeros_like(y)


    final_idx = []
    inv_indexs = []
    for i in range(kmeans_clusters):
        inv_indexs.append(np.where(kmeans_ind == i)[0])
        tempX = X[inv_indexs[-1]]
        print(f"solving {tempX.shape}")

        if nn_solver == "exact":
            nn = NearestNeighbors(n_jobs=-1, p=sep_measure).fit(tempX)
            ind = nn.radius_neighbors(tempX, radius=eps, return_distance=False)
        #elif nn_solver == "faiss":
        #    index = faiss.IndexLSH(d, solver_kwargs['n_bits'])
        #    index.add(X)
        #    _, ind = self.index.search(X, )
        else:
            raise ValueError(f"Not supported nn_solver, {nn_solver}")

        G = nx.Graph()
        G.add_nodes_from(list(range(len(tempX))))
        for i in range(len(tempX)):
            for j in ind[i]:
                if y[i] != y[j]:
                    G.add_edge(i, j)
        idx = min_weighted_vertex_cover(G)
        idx = list(set(range(len(tempX))) - idx)

        final_idx += list(inv_indexs[-1][idx])

    X = X.reshape([len(X)] + ori_shape[1:])

    return X[final_idx], y[final_idx]

"""
Approximated adversarial pruning
"""
from __future__ import division
import logging

import networkx as nx
from networkx.algorithms.approximation.vertex_cover import min_weighted_vertex_cover
import numpy as np
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import cdist

logger = logging.getLogger(__name__)

def approx_ap(X, y, eps, sep_measure):
    ori_shape = list(X.shape)
    X = X.reshape((len(X), -1))
    nn = NearestNeighbors(n_jobs=-1, p=sep_measure).fit(X)
    ind = nn.radius_neighbors(X, radius=eps, return_distance=False)
    G = nx.Graph()
    G.add_nodes_from(list(range(len(X))))
    for i in range(len(X)):
        for j in ind[i]:
            if y[i] != y[j]:
                G.add_edge(i, j)
    idx = min_weighted_vertex_cover(G)
    idx = list(set(range(len(X))) - idx)
    X = X.reshape([len(X)] + ori_shape[1:])
    return X[idx], y[idx]

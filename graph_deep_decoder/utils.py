import numpy as np
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.sparse.csgraph import dijkstra
from scipy import sparse
import matplotlib.pyplot as plt
from pygsp.graphs import Graph, StochasticBlockModel, ErdosRenyi


"""
Distribute contiguous nodes in the same community while assuring that all
communities have (approximately) the same number of nodes.
"""


def bandlimited_model(x_n, V, n_coefs=63, max_coefs=True):
    x_f = np.matmul(np.transpose(V), x_n)
    if max_coefs:
        max_indexes = np.argsort(-np.abs(x_f))[:n_coefs]
        return np.matmul(V[:, max_indexes], x_f[max_indexes])
    return np.matmul(V[:, 0:n_coefs], x_f[0:n_coefs])

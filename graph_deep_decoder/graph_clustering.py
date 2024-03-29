from enum import Enum

import matplotlib.pyplot as plt
import numpy as np
from pygsp.graphs import Graph
from scipy.cluster.hierarchy import dendrogram, fcluster, linkage
from scipy.sparse.csgraph import dijkstra


class Type_A(Enum):
    NONE = 0
    BIN = 1
    WEI = 2


# Error constants
ERR_NON_DEC_SIZE = "{} is not a valid size. All sizes must be non-decreasing"
ERR_DIFF_SIZE = "Last number of clusters ({}) must match graph size ({})"
ERR_UNK_UPS = 'Unkown type of A: {}'


# TODO: use only matrix U!!
class MultiResGraphClustering():
    """
    This class computes a bottom-up multiresolution hierarchichal clustering
    of the given graph. An adjacency matrix may be estimated for each
    resolution level based on the relations of the nodes grouped in the
    different clusters. Additionaly, upsampling and downsampling matrices
    may be estimated for going from different levels of the hierarchy.
    """
    # k represents the size of the root cluster
    def __init__(self, G, n_clusts, k, algorithm='spectral_clutering',
                 method='maxclust', link_fun='average', type_A=Type_A.WEI):

        assert isinstance(type_A, Type_A) or type_A is None, \
            ERR_UNK_UPS.format(type_A)

        self.G = G
        self.sizes = []
        self.cluster_alg = getattr(self, algorithm)
        self.k = k
        self.link_fun = link_fun
        self.labels = []
        self.Z = None
        self.As = []
        self.descendances = []
        self.Us = []

        # Check correct sizes
        for i in range(1, len(n_clusts)):
            if n_clusts[i-1] > n_clusts[i]:
                raise RuntimeError(ERR_NON_DEC_SIZE.format(n_clusts))
        assert n_clusts[-1] == G.N, ERR_DIFF_SIZE.format(n_clusts[-1], G.N)

        non_rep_sizes = list(dict.fromkeys(n_clusts))
        if len(non_rep_sizes) == 1:
            self.sizes = n_clusts
            return

        self.compute_clusters(n_clusts, method)
        self.compute_Ups()
        self.compute_hierarchy_A(type_A)

    def distance_clustering(self):
        """
        Obtain the matrix Z of distances between the different agglomeartive
        clusters by using the distance of Dijkstra among the nodes of the graph
        as a meassure of similarity
        """
        D = dijkstra(self.G.W)
        D = D[np.triu_indices_from(D, 1)]
        self.Z = linkage(D, self.link_fun)

    def spectral_clutering(self):
        """
        Obtain the matrix Z of distances between the different agglomeartive
        clusters by using the first k eigenvectors of the Laplacian matrix as
        node embeding
        """
        self.G.compute_laplacian()
        self.G.compute_fourier_basis()
        X = self.G.U[:, 1:self.k]
        self.Z = linkage(X, self.link_fun)

    def compute_clusters(self, n_clusts, method):
        self.cluster_alg()
        for i, t in enumerate(n_clusts):
            if i > 0 and t == n_clusts[i-1]:
                self.sizes.append(t)
                # NUEVO --> igual borrar!
                self.labels.append(self.labels[-1])
                continue
            if t == self.G.N:
                self.labels.append(np.arange(1, self.G.N+1))
                self.sizes.append(self.G.N)
                continue
            # t represent de relative distance, so it is necessary to
            # obtain the # real desired distance
            if method == 'distance':
                t = t*self.Z[-self.k, 2]
            level_labels = fcluster(self.Z, t, criterion=method)
            self.labels.append(level_labels)
            self.sizes.append(np.unique(level_labels).size)

    def plot_dendrogram(self, show=True):
        plt.figure()
        dendrogram(self.Z, orientation='left', no_labels=False)
        plt.gca().tick_params(labelsize=16)
        if show:
            plt.show()

    # TODO: check if this is really needed!
    def compute_hierarchy_descendance(self):
        # Compute descendance only when there is a change of size
        for i in range(len(self.sizes)-1):
            self.descendances.append([])
            if self.sizes[i] == self.sizes[i+1]:
                continue

            # Find parent (labels i) of each child cluster (labels i+1)
            for j in range(self.sizes[i+1]):
                indexes = np.where(self.labels[i+1] == j+1)
                # Check if all has the same value!!!
                n_parents = np.unique(self.labels[i+1][indexes]).size
                if n_parents != 1:
                    raise RuntimeError("child {} belong to {} parents"
                                       .format(j, n_parents))

                parent_id = self.labels[i][indexes][0]
                self.descendances[i].append(parent_id)
        return self.descendances

    def compute_Ups(self):
        """
        Compute upsampling matrices Us
        """
        self.compute_hierarchy_descendance()
        for i in range(len(self.descendances)):
            if not self.descendances[i]:
                self.Us.append(None)
                continue
            descendance = np.asarray(self.descendances[i])
            U = np.zeros((self.sizes[i+1], self.sizes[i]))
            for j in range(self.sizes[i+1]):
                U[j, descendance[j]-1] = 1
            self.Us.append(U)

    # TODO: review method
    def compute_hierarchy_A(self, A_type):
        if A_type is None or A_type is Type_A.NONE:
            return

        A = self.G.W.todense()
        for i in range(1, len(self.sizes)):
            if self.sizes[i] == self.sizes[i-1]:
                self.As.append(None)
                continue
            N = self.sizes[i]
            self.As.append(np.zeros((N, N)))

            inter_clust_links = 0
            for j in range(N-1):
                nodes_c1 = np.where(self.labels[i] == j+1)[0]
                for k in range(j+1, N):
                    nodes_c2 = np.where(self.labels[i] == k+1)[0]
                    sub_A = A[nodes_c1, :][:, nodes_c2]

                    if A_type == Type_A.BIN and np.sum(sub_A) > 0:
                        self.As[i-1][j, k] = self.As[i-1][k, j] = 1
                    if A_type == Type_A.WEI:
                        self.As[i-1][j, k] = np.sum(sub_A)
                        self.As[i-1][k, j] = self.As[i-1][j, k]
                        inter_clust_links += np.sum(sub_A)
            if A_type == Type_A.WEI:
                if inter_clust_links == 0:
                    print('WARNING: disconnected clusters.')
                else:
                    self.As[i-1] = self.As[i-1]/inter_clust_links
        return self.As

    def plot_labels(self, n_plots=None, show=True):
        self.G.set_coordinates()
        n_labels = len(self.labels)
        n_plots = min(n_plots, n_labels) if n_plots else n_labels
        for i in range(n_plots):
            label = self.labels[i]
            plt.figure()
            self.G.plot_signal(label)
            plt.title('Clusters:' + str(self.sizes[i]))

        if show:
            plt.show()

    def plot_As(self, show=True):
        _, axes = plt.subplots(2, len(self.As))
        for i in range(len(self.As)):
            G = Graph(self.As[i])
            G.set_coordinates('spring')
            axes[0, i].spy(self.As[i])
            G.plot(ax=axes[1, i])
        if show:
            plt.show()

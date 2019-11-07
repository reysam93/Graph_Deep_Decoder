import numpy as np
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.sparse.csgraph import dijkstra
import matplotlib.pyplot as plt
from pygsp.graphs import Graph


# Upsampling Method Constants
NONE = 0
REG = 1
NO_A = 2
BIN = 3
WEI = 4


class MultiResGraphClustering():
    """
    This class computes a bottom-up multiresolution hierarchichal clustering
    of the given graph. An adjacency matrix may be estimated for each
    resolution level based on the relations of the nodes grouped in the
    different clusters. Additionaly, upsampling and downsampling matrices
    may be estimated for going from different levels of the hierarchy.
    """
    # k represents the size of the root cluster
    # TODO: if all have same size what happens??
    def __init__(self, G, n_clusts, k, algorithm='spectral_clutering',
                 method='maxclust', link_fun='average', up_method=WEI):

        # Check non-decreasing sizes
        for i in range(1, len(n_clusts)):
            if n_clusts[i-1] > n_clusts[i]:
                raise RuntimeError("{} is not a valid size. All sizes must be non-decreasing"
                                   .format(n_clusts))

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

        self.compute_clusters(n_clusts, method)
        self.compute_hierarchy_A(up_method)
        self.compute_Ups()

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
        sizes = list(dict.fromkeys(self.sizes))
        for i in range(len(sizes)-1):
            self.descendances.append([])
            # Find parent (labels i) of each child cluster (labels i+1)
            for j in range(sizes[i+1]):
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
        sizes = list(dict.fromkeys(self.sizes))
        self.compute_hierarchy_descendance()
        for i in range(len(self.descendances)):
            descendance = np.asarray(self.descendances[i])
            U = np.zeros((sizes[i+1], sizes[i]))
            for j in range(sizes[i+1]):
                U[j, descendance[j]-1] = 1
            self.Us.append(U)

    def compute_hierarchy_A(self, up_method):
        if up_method in [None, NO_A, REG]:
            return

        A = self.G.W.todense()
        sizes = list(dict.fromkeys(self.sizes))
        for i in range(len(sizes)):
            N = sizes[i]
            self.As.append(np.zeros((N, N)))

            # Normalize matrix A in here??
            inter_clust_links = 0
            for j in range(N-1):
                nodes_c1 = np.where(self.labels[i] == j+1)[0]
                for k in range(j+1, N):
                    nodes_c2 = np.where(self.labels[i] == k+1)[0]
                    sub_A = A[nodes_c1, :][:, nodes_c2]

                    if up_method == BIN and np.sum(sub_A) > 0:
                        self.As[i][j, k] = self.As[i][k, j] = 1
                    if up_method == WEI:
                        self.As[i][j, k] = np.sum(sub_A)
                        self.As[i][k, j] = self.As[i][j, k]
                        inter_clust_links += np.sum(sub_A)
            if up_method == WEI:
                self.As[i] = self.As[i]/inter_clust_links
        return self.As

    def plot_labels(self, show=True):
        n_labels = len(self.labels)
        _, axes = plt.subplots(1, n_labels)
        self.G.set_coordinates()
        for i in range(n_labels):
            self.G.plot_signal(self.labels[i], ax=axes[i])
        if show:
            plt.show()

    def plot_As(self, show=True):
        _, axes = plt.subplots(2, len(self.As))
        for i in range(len(self.As)):
            G = Graph(self.As[i])
            G.set_coordinates()
            axes[0, i].spy(self.As[i])
            G.plot(ax=axes[1, i])
        if show:
            plt.show()

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
def assign_nodes_to_comms(N,k):
    z = np.zeros(N, dtype=np.int)
    leftover_nodes = N % k
    grouped_nodes = 0
    for i in range(k):
        if leftover_nodes > 0:
            n_nodes = np.ceil(N/k).astype(np.int)
            leftover_nodes -= 1
        else:
            n_nodes = np.floor(N/k).astype(np.int)
        z[grouped_nodes:(grouped_nodes+n_nodes)] = i
        grouped_nodes += n_nodes
    return z

def plot_graph_clusters(G, labels, n_clusts):
    _, axes = plt.subplots(1, n_clusts)
    G.set_coordinates(kind='community2D')
    for i in range(n_clusts):
        G.plot_signal(labels[i], ax=axes[i])
    plt.show()

def create_graph(ps, seed=None, type_z='alternated'):
    if ps['type'] == 'SBM':
        if type_z == 'contiguos':
            z = assign_nodes_to_comms(ps['N'],ps['k'])
        elif type_z == 'alternated':
            z = np.array(list(range(ps['k']))*int(ps['N']/ps['k'])+list(range(ps['N']%ps['k'])))
        else:
            z = None
        return StochasticBlockModel(N=ps['N'], k=ps['k'], p=ps['p'], z=z,
                                    q=ps['q'], connected=True, seed=seed)
    elif ps['type'] == 'ER':
        return ErdosRenyi(N=ps['N'], p=ps['p'], connected=True, seed=seed)
    else:
        raise RuntimeError('Unknown graph type')

class DifussedSparseGraphSignal():
    def  __init__(self, G, L, n_deltas, min_d=-1, max_d=1):
        self.G = G
        self.n_deltas = n_deltas
        self.random_sparse_s(min_d, max_d)
        self.random_diffusing_filter(L)
        self.x = np.asarray(self.H.dot(self.s))
        
    """
    Create random sparsee signal s composed of different deltas placed in the different
    communities of the graph, which is supposed to follow an SBM
    """
    def random_sparse_s(self, min_delta, max_delta):
        # Create delta values
        step = (max_delta-min_delta)/(self.n_deltas-1)
        delta_means = np.arange(min_delta, max_delta+0.1, step)
        delta_values = np.random.randn(self.n_deltas)*step/4 + delta_means

        # Randomly assign delta value to comm nodes
        self.s = np.zeros((self.G.W.shape[0]))
        for delta in range(self.n_deltas):
            comm_i = delta % self.G.info['comm_sizes'].size
            comm_nodes, = np.asarray(self.G.info['node_com']==comm_i).nonzero()
            rand_index = np.random.randint(0,self.G.info['comm_sizes'][comm_i])
            selected_node = comm_nodes[rand_index]
            self.s[selected_node] = delta_values[comm_i]

    """
    Create a random diffusing filter with L random coefficients  
    """
    def random_diffusing_filter(self, L):
        # NOTE: One filter or one per comm??
        hs = np.random.rand(L)
        self.H = np.zeros(self.G.W.shape)
        S = self.G.W.todense()
        for l in range(L):
            self.H += hs[l]*np.linalg.matrix_power(S,l)
        #self.H = self.H/np.linalg.norm(self.H,'fro')

    def signal_to_0_1_interval(self):
        self.x -= np.amin(self.x)
        self.x = self.x / np.amax(self.x)

    def normalize(self):
        self.x = (self.x - np.mean(self.x))/np.std(self.x)

    def to_unit_norm(self):
        self.x = self.x/np.linalg.norm(self.x)

# NOTE: maybe should use inheritance instead of selecting the 'algorithm'?
# NOTE: maybe descendance and hier_A should boh be lists, no dictionaries
# NOTE: maybe a static member as a factory method (@staticmethod and without self) 
"""
"""
class MultiRessGraphClustering():
    # k represents the size of the root cluster
    def __init__(self, G, n_clust, k, algorithm='spectral_clutering', 
                    method='maxclust', link_fun='average'):
        self.G = G
        # self.clusters_size = n_clust
        self.clusters_size = []
        self.cluster_alg = getattr(self, algorithm)
        self.k = k
        self.link_fun = link_fun
        self.labels = []
        self.Z = None
        self.descendance = {}
        self.hier_A = []
        self.cluster_alg(G)
        
        for t in n_clust[:-1]:
            # t represent de relative distance, so it is necessary to obtain the 
            # real desired distance
            if method == 'distance':
                t = t*self.Z[-k,2]
            level_labels = fcluster(self.Z, t, criterion=method)
            self.labels.append(level_labels)
            self.clusters_size.append(np.unique(level_labels).size)
        self.labels.append(np.arange(1,G.W.shape[0]+1))
        self.clusters_size.append(G.W.shape[0])

    def distance_clustering(self, G):
        D = dijkstra(G.W)
        D = D[np.triu_indices_from(D,1)]
        self.Z = linkage(D,self.link_fun)

    def spectral_clutering(self, G):
        G.compute_laplacian()
        G.compute_fourier_basis()
        X = G.U[:,1:self.k]
        self.Z = linkage(X, self.link_fun)

    def plot_dendrogram(self):
        plt.figure()
        dendrogram(self.Z)
        plt.show()

    def compute_hierarchy_descendance(self):
        for i in range(len(self.clusters_size)-1):
            self.descendance[i] = []
            # Find parent (labels i) of each child cluster (labels i+1)
            for j in range(self.clusters_size[i+1]):
                indexes = np.where(self.labels[i+1] == j+1)
                # Check if all has the same value!!!
                n_parents = np.unique(self.labels[i+1][indexes]).size
                if n_parents != 1:
                    raise RuntimeError("child {} belong to {} parents".format(j,n_parents))

                parent_id = self.labels[i][indexes][0]
                self.descendance[i].append(parent_id)

        return self.descendance

    def compute_hierarchy_A(self, up_method):
        if up_method == 'no_A' or up_method == None or up_method == 'original':
            return

        A = self.G.W.todense()
        for i in range(len(self.clusters_size)):
            N = self.clusters_size[i]
            self.hier_A.append(np.zeros((N, N)))

            for j in range(N-1):
                nodes_c1 = np.where(self.labels[i] == j+1)[0]
                for k in range(j+1,N):
                    nodes_c2 = np.where(self.labels[i] == k+1)[0]
                    sub_A = A[nodes_c1,:][:,nodes_c2]

                    if up_method == 'binary' and np.sum(sub_A) > 0:
                        self.hier_A[i][j,k] = self.hier_A[i][k,j] = 1
                    if up_method == 'weighted':
                        self.hier_A[i][j,k] = np.sum(sub_A)/(len(nodes_c1)*len(nodes_c2))
                        self.hier_A[i][k,j] = self.hier_A[i][j,k]
        return self.hier_A

    def plot_labels(self):
        n_labels = len(self.labels)-1
        _, axes= plt.subplots(1, n_labels)
        self.G.set_coordinates()
        for i in range(n_labels):
            self.G.plot_signal(self.labels[i], ax=axes[i])
        plt.show()

    def plot_hier_A(self):
        _, axes = plt.subplots(2, len(self.hier_A))
        for i in range(len(self.hier_A)):
            G = Graph(self.hier_A[i])
            G.set_coordinates()
            axes[0,i].spy(self.hier_A[i])
            G.plot(ax=axes[1,i])
        plt.show()

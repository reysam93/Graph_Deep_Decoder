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

def create_graph(ps, seed=None, type_z='random'):
    if ps['type'] == 'SBM':
        if type_z == 'contiguos':
            z = assign_nodes_to_comms(ps['N'],ps['k'])
        elif type_z == 'alternated':
            z = np.array(list(range(ps['k']))*int(ps['N']/ps['k'])+list(range(ps['N']%ps['k'])))
        elif type_z == 'random':
            z = assign_nodes_to_comms(ps['N'],ps['k'])
            np.random.shuffle(z)
        else:
            z = None
        
        return StochasticBlockModel(N=ps['N'], k=ps['k'], p=ps['p'], z=z,
                                    q=ps['q'], connected=True, seed=seed)
    elif ps['type'] == 'ER':
        return ErdosRenyi(N=ps['N'], p=ps['p'], connected=True, seed=seed)
    else:
        raise RuntimeError('Unknown graph type')

def bandlimited_model(x_n, V, n_coefs=63, max_coefs=True):
    x_f = np.matmul(np.transpose(V),x_n)
    if max_coefs:
        max_indexes = np.argsort(-np.abs(x_f))[:n_coefs]
        return np.matmul(V[:,max_indexes], x_f[max_indexes])        
    return np.matmul(V[:,0:n_coefs], x_f[0:n_coefs])

class GraphSignal():
    @staticmethod
    def set_seed(seed):
        np.random.seed(seed)

    @staticmethod
    def add_noise(x, n_p):
        x_p = np.square(np.linalg.norm(x))
        return x + np.random.randn(x.size)*np.sqrt(n_p*x_p/x.size)

    @staticmethod
    def generate_inpaint_mask(x, p_miss):
        mask = np.ones(x.size)
        mask[np.random.choice(x.size, int(x.size*p_miss))] = 0
        return mask
    
    # NOTE: make static method for giving objct
    # from desired signal class? --> method create_signal
    def __init__(self, G):
        self.G = G
        self.x = None
        self.x_n = None


    """
    Create a lineal random diffusing filter with L random coefficients  
    """
    def random_diffusing_filter(self, L):
        # NOTE: One filter or one per comm??
        hs = np.random.rand(L)
        self.H = np.zeros(self.G.W.shape)
        S = self.G.W.todense()
        for l in range(L):
            self.H += hs[l]*np.linalg.matrix_power(S,l)

    def median_neighbours_nodes(self):
        x_aux = np.zeros(self.x.shape)
        for i in range(self.G.N):
            _, neighbours = np.asarray(self.G.W.todense()[i,:]!=0).nonzero()
            x_aux[i] = np.median(self.x[neighbours])
        self.x = x_aux

    def mean_neighbours_nodes(self):
        x_aux = np.zeros(self.x.shape)
        for i in range(self.G.N):
            _, neighbours = np.asarray(self.G.W.todense()[i,:]!=0).nonzero()
            x_aux[i] = np.mean(self.x[neighbours])
        self.x = x_aux

    def signal_to_0_1_interval(self):
        min_x = np.amin(self.x)
        if min_x < 0:
            self.x -= np.amin(self.x)
        
        self.x = self.x / np.amax(self.x)

    def normalize(self):
        self.x = (self.x - np.mean(self.x))/np.std(self.x)

    def to_unit_norm(self):
        if np.linalg.norm(self.x) == 0:
            print("WARNING: signal with norm 0")
            return
        self.x = self.x/np.linalg.norm(self.x)

    def plot(self, show=True):
        self.G.set_coordinates(kind='community2D')
        self.G.plot_signal(self.x)
        if show:
            plt.show()

class DeterministicGS(GraphSignal):
    def __init__(self, G, x):
       GraphSignal.__init__(self, G)
       self.x = x 

class DifussedSparseGS(GraphSignal):
    def __init__(self, G, L, n_deltas, min_d=-1, max_d=1):
        GraphSignal.__init__(self, G)
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

class NonLinealDSGS(DifussedSparseGS):
    def __init__(self, G, L, n_deltas, D=None, min_d=-1, max_d=1):
        if D is None:
            self.D = dijkstra(G.W)
        else:
            self.D = D
        DifussedSparseGS.__init__(self,G,L,n_deltas,min_d,max_d)

    """
    Create a non-linear random diffusing filter with L random coefficients.
    Assumes the graph is binary
    """
    def random_diffusing_filter(self, L):
        decay_coeff = 0.9
        self.H = np.zeros(self.G.W.shape)
        for l in range(L):
            L_Neighbours = np.zeros(self.G.W.shape)
            L_Neighbours[self.D==l] = 1
            self.H += np.power(decay_coeff,l)*L_Neighbours

class MedianDSGS(DifussedSparseGS):
    def __init__(self, G, L, n_deltas, min_d=-1, max_d=1):
        DifussedSparseGS.__init__(self,G,L,n_deltas,min_d,max_d)
        self.median_neighbours_nodes()

class MeanMedianDSGS(DifussedSparseGS):
    def __init__(self, G, L, n_deltas, min_d=-1, max_d=1):
        DifussedSparseGS.__init__(self,G,L,n_deltas,min_d,max_d)
        self.mean_neighbours_nodes()
        mean_x = self.x        
        self.median_neighbours_nodes()
        self.x = self.x * mean_x 
        #self.x = self.x*x_aux
        

class NLCombinationsDSGS(DifussedSparseGS):
    def __init__(self, G, L, n_deltas, min_d=-1, max_d=1):
        DifussedSparseGS.__init__(self,G,L,n_deltas,min_d,max_d)
        self.nl_combs = [np.min, np.mean, np.median, np.max]
        self.nl_combinations_comm_nodes()

    def nl_combinations_comm_nodes(self):
        for i in range(self.G.N):
            comm = self.G.info['node_com'][i]
            nonlinearity = self.nl_combs[comm % len(self.nl_combs)]
            _, neighbours = np.asarray(self.G.W.todense()[i,:]!=0).nonzero()
            self.x[neighbours] = nonlinearity(self.x[neighbours])

class NonBLMedian(GraphSignal):
    def __init__(self, G, min_mean=-1, max_mean=1):
        GraphSignal.__init__(self,G)
        self.n_comms = self.G.info['comm_sizes'].size
        self.x_from_freq()
        self.median_neighbours_nodes()

    def x_from_freq(self):
        self.G.compute_fourier_basis()
        x_freq = np.random.uniform(size=self.G.N)
        self.x = np.matmul(self.G.U,x_freq)
        #self.random_diffusing_filter(2)
        #self.x = np.asarray(self.H.dot(self.x))
        self.median_neighbours_nodes()

    #def random_comm_noise(self, min_mean, max_mean):
    #    step = (max_mean-min_mean)/(self.n_comms-1)
    #    comm_means = np.arange(min_mean, max_mean+0.1, step)

    #    self.x = np.zeros((self.G.N))
    #    for comm in range(self.n_comms):
    #        comm_nodes, = np.asarray(self.G.info['node_com']==comm).nonzero()
    #        self.x[comm_nodes] = np.random.randn(comm_nodes.size)*step/4 + comm_means[comm]


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
        dendrogram(self.Z, orientation='left', no_labels=True)
        plt.gca().tick_params(labelsize=16)
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

            inter_clust_links = 0
            for j in range(N-1):
                nodes_c1 = np.where(self.labels[i] == j+1)[0]
                for k in range(j+1,N):
                    nodes_c2 = np.where(self.labels[i] == k+1)[0]
                    sub_A = A[nodes_c1,:][:,nodes_c2]

                    if up_method == 'binary' and np.sum(sub_A) > 0:
                        self.hier_A[i][j,k] = self.hier_A[i][k,j] = 1
                    if up_method == 'weighted':
                        self.hier_A[i][j,k] = np.sum(sub_A)
                        self.hier_A[i][k,j] = self.hier_A[i][j,k]
                        inter_clust_links += np.sum(sub_A)
            if up_method == 'weighted':
                self.hier_A[i] = self.hier_A[i]/inter_clust_links
        return self.hier_A

    def plot_labels(self, show=True):
        n_labels = len(self.labels)-1
        _, axes= plt.subplots(1, n_labels)
        self.G.set_coordinates()
        for i in range(n_labels):
            self.G.plot_signal(self.labels[i], ax=axes[i])
        if show:    
            plt.show()

    def plot_hier_A(self, show=True):
        _, axes = plt.subplots(2, len(self.hier_A))
        for i in range(len(self.hier_A)):
            G = Graph(self.hier_A[i])
            G.set_coordinates()
            axes[0,i].spy(self.hier_A[i])
            G.plot(ax=axes[1,i])
        if show:
            plt.show()

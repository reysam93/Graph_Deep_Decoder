import matplotlib.pyplot as plt
import numpy as np
from pygsp.graphs import (BarabasiAlbert, ErdosRenyi, Graph,
                          StochasticBlockModel)
from scipy.sparse.csgraph import dijkstra

# Signals Type Constants
LINEAR = 1
NON_LINEAR = 2
MEDIAN = 3
COMB = 4
NOISE = 5
NON_BL = 6

# Graph Type Constants
SBM = 1
ER = 2
BA = 3

# Comm Node Assignment Constants
CONT = 1    # Contiguous nodes
ALT = 2    # Alternated nodes
RAND = 3    # Random nodes


def assign_nodes_to_comms(N, k):
    """
    Distribute contiguous nodes in the same community while assuring that all
    communities have (approximately) the same number of nodes.
    """
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


def create_graph(ps, seed=None):
    """
    Create a random graph using the parameters specified in the dictionary ps.
    The keys that this dictionary should nclude are:
        - type: model for the graph. Options are SBM (Stochastic Block Model)
          or ER (Erdos-Renyi)
        - N: number of nodes
        - k: number of communities (for SBM only)
        - p: edge probability for nodes in the same community
        - q: edge probability for nodes in different communities (for SBM only)
        - type_z: specify how to assigns nodes to communities (for SBM only).
          Options are CONT (continous), ALT (alternating) and RAND (random)
    """
    if ps['type'] == SBM:
        if ps['type_z'] == CONT:
            z = assign_nodes_to_comms(ps['N'], ps['k'])
        elif ps['type_z'] == ALT:
            z = np.array(list(range(ps['k']))*int(ps['N']/ps['k']) +
                         list(range(ps['N'] % ps['k'])))
        elif ps['type_z'] == RAND:
            z = assign_nodes_to_comms(ps['N'], ps['k'])
            np.random.shuffle(z)
        else:
            z = None
        G = StochasticBlockModel(N=ps['N'], k=ps['k'], p=ps['p'], z=z,
                                 q=ps['q'], connected=True, seed=seed,
                                 max_iter=20)
        G.set_coordinates('community2D')
        return G
    elif ps['type'] == ER:
        G = ErdosRenyi(N=ps['N'], p=ps['p'], connected=True, seed=seed,
                       max_iter=20)
        G.set_coordinates('community2D')
        return G
    elif ps['type'] == BA:
        G = BarabasiAlbert(N=ps['N'], m=ps['m'], m0=ps['m0'], seed=seed)
        G.info = {'comm_sizes': np.array([ps['N']]),
                  'node_com': np.zeros((ps['N'],), dtype=int)}
        G.set_coordinates('spring')
        return G
    else:
        raise RuntimeError('Unknown graph type')


class GraphSignal():
    @staticmethod
    def create_graph_signal(signal_type, G, L, k, D=None):
        if signal_type == LINEAR:
            signal = DifussedSparseGS(G, L, k)
        elif signal_type == NON_LINEAR:
            signal = NonLinealDSGS(G, L, k, D)
        elif signal_type == MEDIAN:
            signal = MedianDSGS(G, L, k)
        elif signal_type == COMB:
            signal = NLCombinationsDSGS(G, L, k)
        elif signal_type == NOISE:
            signal = DeterministicGS(G, np.random.randn(G.N))
        elif signal_type == NON_BL:
            signal = NonBLMedian(G)
        else:
            raise RuntimeError('Unknown signal type')
        return signal

    @staticmethod
    def add_noise(x, n_p):
        if n_p == 0:
            return x
        x_p = np.square(np.linalg.norm(x))
        return x + np.random.randn(x.size)*np.sqrt(n_p*x_p/x.size)

    @staticmethod
    def generate_inpaint_mask(x, p_miss):
        mask = np.ones(x.size)
        mask[np.random.choice(x.size, int(x.size*p_miss))] = 0
        return mask

    def __init__(self, G):
        self.G = G
        self.x = None
        self.x_n = None

    def median_neighbours_nodes(self):
        x_aux = np.zeros(self.x.shape)
        for i in range(self.G.N):
            _, neighbours = np.asarray(self.G.W.todense()[i, :] != 0).nonzero()
            x_aux[i] = np.median(self.x[np.append(neighbours, i)])
        self.x = x_aux

    def mean_neighbours_nodes(self):
        x_aux = np.zeros(self.x.shape)
        for i in range(self.G.N):
            _, neighbours = np.asarray(self.G.W.todense()[i, :] != 0).nonzero()
            x_aux[i] = np.mean(self.x[np.append(neighbours, i)])
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

    def random_diffusing_filter(self, L):
        """
        Create a lineal random diffusing filter with L random coefficients
        """
        hs = np.random.rand(L)
        self.H = np.zeros(self.G.W.shape)
        S = self.G.W.todense()
        for l in range(L):
            self.H += hs[l]*np.linalg.matrix_power(S, l)

    def random_sparse_s(self, min_delta, max_delta):
        """
        Create random sparse signal s composed of deltas placed in the
        different communities of the graph if more than one community
        exists. Otherwise, deltas are randomly placed.
        """
        # Create delta values
        n_comms = self.G.info['comm_sizes'].size
        if n_comms > 1:
            step = (max_delta-min_delta)/(n_comms-1)
        else:
            step = (max_delta-min_delta)/(self.n_deltas-1)

        ds_per_comm = np.ceil(self.n_deltas/n_comms).astype(int)
        delta_means = np.arange(min_delta, max_delta+0.1, step)
        delta_means = np.tile(delta_means, ds_per_comm)[:self.n_deltas]
        delta_vals = np.random.randn(self.n_deltas)*step/4 + delta_means

        # Randomly assign delta value to comm nodes
        self.s = np.zeros((self.G.N))
        for delta in range(self.n_deltas):
            comm = delta % n_comms
            comm_nodes, = np.asarray(self.G.info['node_com'] == comm).nonzero()
            rand_index = np.random.randint(0, self.G.info['comm_sizes'][comm])
            selected_node = comm_nodes[rand_index]
            self.s[selected_node] = delta_vals[comm]

    def plot(self, show=True):
        _, axes = plt.subplots(1, 2)
        self.G.plot_signal(self.s, ax=axes[0])
        self.G.plot_signal(self.x, ax=axes[1])
        if show:
            plt.show()


class NonLinealDSGS(DifussedSparseGS):
    def __init__(self, G, L, n_deltas, D=None, min_d=-1, max_d=1):
        if D is None:
            self.D = dijkstra(G.W)
        else:
            self.D = D
        DifussedSparseGS.__init__(self, G, L, n_deltas, min_d, max_d)

    """
    Create a non-linear random diffusing filter with L random coefficients.
    Assumes the graph is binary
    """
    def random_diffusing_filter(self, L):
        decay_coeff = 0.9
        self.H = np.zeros(self.G.W.shape)
        for l in range(L):
            L_Neighbours = np.zeros(self.G.W.shape)
            L_Neighbours[self.D == l] = 1
            self.H += np.power(decay_coeff, l)*L_Neighbours


class MedianDSGS(DifussedSparseGS):
    def __init__(self, G, L, n_deltas, min_d=-1, max_d=1):
        DifussedSparseGS.__init__(self, G, L, n_deltas, min_d, max_d)
        self.median_neighbours_nodes()


class MeanMedianDSGS(DifussedSparseGS):
    def __init__(self, G, L, n_deltas, min_d=-1, max_d=1):
        DifussedSparseGS.__init__(self, G, L, n_deltas, min_d, max_d)
        self.mean_neighbours_nodes()
        mean_x = self.x
        self.median_neighbours_nodes()
        self.x = self.x * mean_x
        # self.x = self.x*x_aux


class NLCombinationsDSGS(DifussedSparseGS):
    def __init__(self, G, L, n_deltas, min_d=-1, max_d=1):
        DifussedSparseGS.__init__(self, G, L, n_deltas, min_d, max_d)
        self.nl_combs = [np.min, np.mean, np.median, np.max]
        self.nl_combinations_comm_nodes()

    def nl_combinations_comm_nodes(self):
        for i in range(self.G.N):
            comm = self.G.info['node_com'][i]
            nonlinearity = self.nl_combs[comm % len(self.nl_combs)]
            _, neighbours = np.asarray(self.G.W.todense()[i, :] != 0).nonzero()
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
        # self.random_diffusing_filter(2)
        # self.x = np.asarray(self.H.dot(self.x))
        self.median_neighbours_nodes()

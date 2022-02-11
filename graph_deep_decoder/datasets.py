from enum import Enum

import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
from pygsp.graphs import (BarabasiAlbert, ErdosRenyi, Graph,
                          StochasticBlockModel)
from scipy.sparse.csgraph import dijkstra
from sympy import N

from graph_deep_decoder import utils


# Signals Type Constants
class SigType(Enum):
    NOISE = 1   # White Noise
    DS = 2      # Diffused Sparse
    DW = 3      # Diffused White Noise
    SM = 4      # Smooth
    NON_BL = 5   # Non Bandlimited


class NonLin(Enum):
    NONE = 0
    MEDIAN = 1
    SQUARE = 2
    MIN_MAX = 3
    SQ_MED = 4
    SQ2 = 5


# Graph Type Constants
SBM = 1     # Stochastic Block Model
ER = 2      # Erdos Renyi
BA = 3      # Barabasi Albert
SW = 4      # Small World
REG = 5     # Regular Graph
PLC = 6     # Power Law Cluster
CAVE = 7    # Caveman graph

MAX_RETRIES = 25

# Comm Node Assignment Constants
CONT = 1    # Contiguous nodes
ALT = 2     # Alternated nodes
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


# TODO: migrate to networkx library
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
                                 max_iter=MAX_RETRIES)
    elif ps['type'] == ER:
        G = ErdosRenyi(N=ps['N'], p=ps['p'], connected=True, seed=seed,
                       max_iter=MAX_RETRIES)
    elif ps['type'] == BA:
        G = BarabasiAlbert(N=ps['N'], m=ps['m'], m0=ps['m0'], seed=seed)
        G.info = {'comm_sizes': np.array([ps['N']]),
                  'node_com': np.zeros((ps['N'],), dtype=int)}
    elif ps['type'] == SW:
        # ps['k'] < 1 means the proportion of desired links is indicated
        k = ps['k']*(ps['N']-1) if ps['k'] < 1 else ps['k']
        G = nx.connected_watts_strogatz_graph(n=ps['N'], k=int(k), p=ps['p'],
                                              tries=MAX_RETRIES, seed=seed)
        A = nx.to_numpy_array(G)
        G = Graph(A)
    elif ps['type'] == REG:
        # ps['d'] < 1 means the proportion of desired links is indicated
        d = ps['d']*(ps['N']-1) if ps['d'] < 1 else ps['d']
        G = nx.random_regular_graph(n=ps['N'], d=int(d), seed=seed)
        A = nx.to_numpy_array(G)
        G = Graph(A)
    elif ps['type'] == PLC:
        # ps['m'] < 1 means the proportion of desired links is indicated
        m = ps['m']*(ps['N']-1) if ps['m'] < 1 else ps['m']
        G = nx.powerlaw_cluster_graph(n=ps['N'], m=int(m), p=ps['p'],
                                      seed=seed)
        A = nx.to_numpy_array(G)
        G = Graph(A)
    elif ps['type'] == CAVE:
        k = int(ps['N']/ps['l'])
        G = nx.connected_caveman_graph(ps['l'], k=k)
        A = nx.to_numpy_array(G)
        G = Graph(A)
    else:
        raise RuntimeError('Unknown graph type')

    assert G.is_connected(), 'Graph is not connected'

    G.set_coordinates('spring')
    # G.compute_fourier_basis()
    return G


def bandlimited_signal(Lambda, V, p=5, first_p=True, energy=1):
    if p >= 1 and type(p) is int:
        nparams = p
    elif 0 < p and 1 > p:
        nparams = int(p*V.shape[0])
    else:
        return None

    xp = np.random.randn(nparams)
    xp /= np.linalg.norm(xp)
    idx = np.flip(np.argsort(np.abs(Lambda)), axis=0)

    if first_p:
        idx = idx[:nparams]
    else:
        idx = idx[-nparams:]

    x = V[:, idx].dot(xp)

    if energy == 1:
        return x
    elif energy >= 0 and energy < 1:
        noise = np.random.randn(V.shape[0])
        noise /= np.linalg.norm(noise)
        return energy*x + (1-energy)*noise
    else:
        return None


class GraphSignal():
    @staticmethod
    def create(signal_type, G, non_lin,
               L=6, deltas=4, unit_norm=True, center=False,
               to_0_1=False, D=None, coefs=None, pos_coefs=True):
        if signal_type is SigType.NOISE:
            signal = DeterministicGS(G, np.random.randn(G.N))
        elif signal_type is SigType.DS:
            signal = DiffusedSparseGS(G, non_lin, L, deltas, pos_coefs)
        elif signal_type is SigType.DW:
            signal = DiffusedWhiteGS(G, non_lin, L, coefs, pos_coefs)
        elif signal_type is SigType.SM:
            signal = SmoothGS(G, non_lin)
        elif signal_type is SigType.NON_BL:
            signal = NonBandLimited(G, non_lin)
        else:
            raise RuntimeError('Unknown signal type')

        if to_0_1:
            signal.to_0_1_interval()
        if center:
            signal.center()
        if unit_norm:
            signal.to_unit_norm()
        return signal

    @staticmethod
    def add_noise(x, n_p, n_type='gaussian'):
        if n_p == 0:
            return x

        x_p = np.square(np.linalg.norm(x))
        if n_type == 'gaussian':
            return x + np.random.randn(x.size)*np.sqrt(n_p*x_p/x.size)
        elif n_type == 'uniform':
            noise = np.random.rand(x.size)
            noise *= x_p*n_p/np.linalg.norm(noise, 2)
            return x + noise
        else:
            raise Exception('Unkown noise type')

    @staticmethod
    def add_bernoulli_noise(x, n_p):
        if n_p == 0:
            return x

        noise_mask = np.random.rand(x.shape[0]) < n_p
        return np.logical_xor(x, noise_mask)

    @staticmethod
    def change_labels(x, r):
        if r == 0:
            return x

        mask = np.random.rand(x.shape[0]) < r
        return np.where(mask, np.random.randint(x.max()+1, size=x.shape), x)

    @staticmethod
    def generate_inpaint_mask(x, p_miss):
        mask = np.ones(x.size)
        mask[np.random.choice(x.size, int(x.size*p_miss), replace=False)] = 0
        return mask

    def __init__(self, G):
        self.G = G
        self.x = None
        self.x_n = None

    def diffusing_filter(self, L, coefs, pos_coefs):
        """
        Create a lineal random diffusing filter with L random coefficients
        """
        if coefs is not None:
            hs = coefs
        elif pos_coefs:
            hs = np.random.rand(L)  # Uniform [0, 1]
        else:
            hs = np.random.rand(L)*2 - 1  # Uniform [-1, 1]
        self.H = np.zeros(self.G.W.shape)
        S = self.G.W.todense()
        for l in range(L):
            self.H += hs[l]*np.linalg.matrix_power(S, l)

    def apply_non_linearity(self, nl_type):
        if nl_type is NonLin.NONE or nl_type is None:
            return

        x_aux = np.zeros(self.x.shape)
        S = self.G.W.todense() + np.eye(self.G.N)
        for i in range(self.G.N):
            _, neighbours_ind = np.asarray(S[i, :] != 0).nonzero()
            neighbours = self.x[neighbours_ind]
            if nl_type is NonLin.MEDIAN:
                x_aux[i] = np.median(neighbours)
            elif nl_type in [NonLin.SQUARE, NonLin.SQ_MED, NonLin.SQ2]:
                x_aux[i] = np.sum(np.sign(neighbours)*(neighbours**2))
            elif nl_type is NonLin.MIN_MAX:
                x_aux[i] = neighbours[np.argmax(np.abs(neighbours))]
            else:
                print('Unkown non-linearity.')
                return

        self.x = x_aux

        if nl_type is NonLin.SQ_MED:
            self.apply_non_linearity(NonLin.MEDIAN)
        elif nl_type is NonLin.SQ2:
            self.apply_non_linearity(NonLin.SQUARE)

    def to_0_1_interval(self):
        min_x = np.amin(self.x)
        if min_x < 0:
            self.x -= np.amin(self.x)
        self.x = self.x / np.amax(self.x)

    def center(self):
        self.x = self.x - np.mean(self.x)

    def to_unit_norm(self):
        if np.linalg.norm(self.x) == 0:
            print("WARNING: signal with norm 0")
            return
        self.x = self.x/np.linalg.norm(self.x)

    def plot(self, show=True):
        self.G.plot_signal(self.x)
        if show:
            plt.show()

    def smoothness(self):
        """
        Return the smoothness of the signal with respect to the graph.
        """
        return self.x.T.dot(self.G.L.dot(self.x))

    def total_variation(self, norm=1):
        A = self.G.W.toarray()
        eig_vals, _ = np.linalg.eig(A)
        A = A/np.max(np.abs(eig_vals))
        return np.linalg.norm(self.x-A.dot(self.x), norm)**norm

    def check_bandlimited(self, coefs=0.1):
        n_coefs = int(self.G.N*coefs)
        x_freq = self.G.U.T.dot(self.x)
        return np.sum(np.sort(-np.abs(x_freq))[:n_coefs]**2)/np.sum(x_freq**2)

    def check_bl_err(self, coefs=0.1, firsts=True):
        k = int(self.G.N*coefs)
        Lambda, V = utils.ordered_eig(self.G.W.todense())
        V = np.asarray(V)
        x = np.asarray(self.x)
        if firsts:
            x_bl = V[:, :k].dot(V[:, :k].T.dot(x))
        else:
            x_freq = V.T.dot(x)
            idx = np.flip(np.abs(x_freq).argsort(), axis=0)[:k]
            x_bl = V[:, idx].dot(x_freq[idx])

        return np.linalg.norm(x-x_bl)**2

    def plot_freq_resp(self, show=True):
        """
        Compute and plot the frequency response of the graph signal.
        """
        x_freq = self.G.U.T.dot(self.x)
        _, axes = plt.subplots(1, 2)
        axes[0].stem(x_freq)
        self.G.plot_signal(x_freq, ax=axes[1])
        if show:
            plt.show()


class DeterministicGS(GraphSignal):
    def __init__(self, G, x):
        GraphSignal.__init__(self, G)
        self.x = x


class DiffusedSparseGS(GraphSignal):
    def __init__(self, G, non_lin, L, n_deltas,
                 coefs=None, pos_coefs=True, min_d=-1, max_d=1):
        GraphSignal.__init__(self, G)
        self.n_deltas = n_deltas
        self.random_sparse_s(min_d, max_d)
        self.diffusing_filter(L, coefs, pos_coefs)
        self.x = np.asarray(self.H.dot(self.s))
        self.apply_non_linearity(non_lin)

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
            self.s[selected_node] = delta_vals[delta]

    def plot(self, show=True):
        _, axes = plt.subplots(1, 2)
        self.G.plot_signal(self.s, ax=axes[0])
        self.G.plot_signal(self.x, ax=axes[1])
        if show:
            plt.show()


class DiffusedWhiteGS(GraphSignal):
    def __init__(self, G, non_lin, L, coefs=None, pos_coefs=True):
        GraphSignal.__init__(self, G)
        self.s = np.random.randn(G.N)
        self.diffusing_filter(L, coefs, pos_coefs)
        self.x = np.asarray(self.H.dot(self.s))
        self.apply_non_linearity(non_lin)

    def plot(self, show=True):
        _, axes = plt.subplots(1, 2)
        self.G.plot_signal(self.s, ax=axes[0])
        self.G.plot_signal(self.x, ax=axes[1])
        if show:
            plt.show()


class SmoothGS(GraphSignal):
    def __init__(self, G, non_lin):
        GraphSignal.__init__(self, G)
        mean = np.zeros(G.N)
        cov = np.linalg.pinv(np.diag(G.e))
        self.s = np.random.multivariate_normal(mean, cov)
        self.x = G.U.dot(self.s)
        self.apply_non_linearity(non_lin)


class NonBandLimited(GraphSignal):
    def __init__(self, G, non_lin):
        GraphSignal.__init__(self, G)
        x_freq = np.random.rand(self.G.N)/2-0.25
        self.x = np.dot(self.G.U, x_freq)
        self.apply_non_linearity(non_lin)

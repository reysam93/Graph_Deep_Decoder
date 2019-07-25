import numpy as np
import matplotlib.pyplot as plt

# Signals Type
LINEAR = 1
NON_LINEAR = 2
MEDIAN = 3
COMB = 4
NOISE = 5
NON_BL = 6

class GraphSignal():
    
    @staticmethod
    def create_graph_signal(signal_type, G, L, k, D):
        if signal_type == LINEAR:
            signal = DifussedSparseGS(G,L,k)
        elif signal_type == NON_LINEAR:
            signal = NonLinealDSGS(G,L,k,D)
        elif signal_type == MEDIAN:
            signal = MedianDSGS(G,L,k)
        elif signal_type == COMB:
            signal = NLCombinationsDSGS(G,L,k)
        elif signal_type == NOISE:
            signal = DeterministicGS(G,np.random.randn(G.N))
        elif signal_type == NON_BL:
            signal = NonBLMedian(G)
        else:
            raise RuntimeError('Unknown signal type')
        return signal

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

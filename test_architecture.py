"""
Check the efect of changing the size of the architecture for controlig the number of 
parameters which represent the signal
"""

import sys
import time
from multiprocessing import Pool 
sys.path.insert(0, 'graph_deep_decoder')
from graph_deep_decoder import utils
from graph_deep_decoder.architecture import GraphDeepDecoder
from pygsp.graphs import StochasticBlockModel, ErdosRenyi

import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn

# Tuning parameters
n_signals = 50
L = 5
n_p = 0.01 # SNR = 1/n_p
batch_norm = True #True
up_method = 'weighted'
upsampling = True
last_act_fun = nn.Tanh() #nn.Sigmoid()

# Constants
SEED = 15
N_CHANS = [[4]*3, [6]*3, [2]*3, [10,5,3], [4], [9], [4]*4, [3]*6, [8,6,4,2]]
N_CLUSTS = [[4,16,64,256]]*4 + [[4,256]]*2 + [[4,16,64,128,256]] + \
             [[4,8,16,32,64,128,256]] + [[4, 8, 16, 64, 256]]
N_SCENARIOS = len(N_CHANS)

def create_graph(params):
    if params['type'] == 'SBM':
        return StochasticBlockModel(N=params['N'], k=params['k'], p=params['p'],
                                    q=params['q'], connected=True, seed=SEED)
    elif params['type'] == 'ER':
        return ErdosRenyi(N=params['N'], p=params['p'], connected=True, seed=SEED)
    else:
        raise RuntimeError('Unknown graph type')

def plot_clusters(G, cluster):
    G.set_coordinates(kind='community2D')
    _, axes = plt.subplots(1, 2)
    G.plot_signal(cluster.labels[0], ax=axes[0])
    axes[1].spy(G.W)
    plt.show()

def compute_clusters(alg):
    assert(len(N_CHANS) == len(N_CLUSTS))
    descendances = []
    hier_As = []
    for i in range(len(N_CHANS)):
        assert(len(N_CHANS[i])+1 == len(N_CLUSTS[i]))
        cluster = utils.MultiRessGraphClustering(G, N_CLUSTS[i], alg)
        descendances.append(cluster.compute_hierarchy_descendance())
        hier_As.append(cluster.compute_hierarchy_A(up_method))
    return descendances, hier_As

def test_architecture(x, descendances, hier_As):
    mse = np.zeros(N_SCENARIOS)
    params = np.zeros(N_SCENARIOS)
    x_n = x + np.random.randn(x.size)*np.sqrt(n_p)

    for i in range(N_SCENARIOS):
        dec = GraphDeepDecoder(descendances[i], hier_As[i], N_CLUSTS[i],
                        n_channels=N_CHANS[i], up_method=up_method, batch_norm=batch_norm,
                        upsampling=upsampling, last_act_fun=last_act_fun)

        dec.build_network()
        _, mse[i] = dec.fit(x_n)
        mse[i] = mse[i]/np.linalg.norm(x)
        params[i] = dec.count_params()

    return mse, params

def print_results(N, mean_mse, params):
    for i in range(N_SCENARIOS):
        print('{}. (CHANS {}, CLUSTS: {}) '.format(i, N_CHANS[i], N_CLUSTS[i]))
        print('\tMean MSE: {}\tParams: {}\tCompression: {}'
                            .format(mean_mse[i], params[i], N/params[i]))


if __name__ == '__main__':
    # Graph parameters
    G_params = {}
    G_params['type'] = 'SBM' # SBM or ER
    G_params['N'] = N = 256
    G_params['k'] = k = 4
    G_params['p'] = 0.15
    G_params['q'] = 0.01/k

    # Cluster Params
    alg = 'spectral_clutering' # spectral_clutering or distance_clustering
    method = 'maxclust'
    
    G = create_graph(G_params)
    descendances, hier_As = compute_clusters(alg)
    
    start_time = time.time()
    mse = np.zeros((n_signals, N_SCENARIOS))
    with Pool() as pool:
        for i in range(n_signals):
            signal = utils.DifussedSparseGraphSignal(G,L,-1,1)
            # signal.signal_to_0_1_interval
            signal.normalize()
            result = pool.apply_async(test_architecture,
                                        args=[signal.x, descendances, hier_As])

        for i in range(n_signals):
            mse[i,:], n_params = result.get()
    
    # Print result:
    print('--- {} minutes ---'.format((time.time()-start_time)/60))
    print_results(N, np.mean(mse, axis=0), n_params)
    
    
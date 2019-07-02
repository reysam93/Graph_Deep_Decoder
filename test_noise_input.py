"""
Check the efect of changing the size of the architecture for controlig the number of 
parameters which represent the signal
"""

import sys, os
import time, datetime
from multiprocessing import Pool 
sys.path.insert(0, 'graph_deep_decoder')
from graph_deep_decoder import utils
from graph_deep_decoder.architecture import GraphDeepDecoder
from pygsp.graphs import StochasticBlockModel, ErdosRenyi

import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn

# Tuning parameters
n_signals = 5
L = 6
batch_norm = True #True
act_fun = nn.ReLU()
up_method = 'weighted'
last_act_fun = nn.Sigmoid() #nn.Tanh()

# Constants
SEED = 15
N_CHANS = [[6]*3, [4]*3, [3]*3, [2]*3, [10,5,3],  [4,3,3], [4], [9], [4]*4, [8,6,4,2],
           [4,3,2,2], [3]*6, [2]*6, [4, 4, 3, 3, 2], [8, 8, 8, 8]]
N_CLUSTS = [[4,16,64,256]]*6 + [[4,256]]*2 + [[4,16,32,64,256]]*3 + \
            [[4,8,16,32,64,128,256]]*2 + [[4, 8, 16 ,32, 64, 256]] + \
            [[4, 16, 32, 64, 256]]
N_SCENARIOS = len(N_CHANS)

def plot_clusters(G, cluster):
    G.set_coordinates(kind='community2D')
    _, axes = plt.subplots(1, 2)
    G.plot_signal(cluster.labels[0], ax=axes[0])
    axes[1].spy(G.W)
    plt.show()

def compute_clusters(alg, k):
    assert(len(N_CHANS) == len(N_CLUSTS))
    sizes = []
    descendances = []
    hier_As = []
    for i in range(len(N_CHANS)):
        assert(len(N_CHANS[i])+1 == len(N_CLUSTS[i]))
        cluster = utils.MultiRessGraphClustering(G, N_CLUSTS[i], k, alg)
        sizes.append(cluster.clusters_size)
        descendances.append(cluster.compute_hierarchy_descendance())
        hier_As.append(cluster.compute_hierarchy_A(up_method))
    return sizes, descendances, hier_As

def test_architecture(id, x, sizes, descendances, hier_As):
    error = np.zeros(N_SCENARIOS)
    mse_fit = np.zeros(N_SCENARIOS)
    params = np.zeros(N_SCENARIOS)
    for i in range(N_SCENARIOS):
        dec = GraphDeepDecoder(descendances[i], hier_As[i], sizes[i],
                        n_channels=N_CHANS[i], upsampling=up_method, batch_norm=batch_norm,
                        last_act_fun=last_act_fun, act_fun=act_fun)

        dec.build_network()
        x_est, mse_fit[i] = dec.fit(x)
        
        error[i] = np.sum(np.square(x-x_est))/np.square(np.linalg.norm(x))
        params[i] = dec.count_params()
        print('Signal: {} Scenario: {} ({} params): Error: {:.4f}'
                            .format(id, i, params[i], error[i]))
    return error, params

def print_results(N, mean_error, params, mean_mse_fit):
    for i in range(N_SCENARIOS):
        print('{}. (CHANS {}, CLUSTS: {}) '.format(i+1, N_CHANS[i], N_CLUSTS[i]))
        print('\tMean MSE: {}\tParams: {}\tCompression: {}\tMEdian MSE: {}'
                            .format(mean_error[i], params[i], N/params[i], mean_mse_fit[i]))

def save_results(error, n_params, G_params):
    if not os.path.isdir('./results/test_input'):
        os.makedirs('./results/test_input')

    data = {'SEED': SEED, 'N_CHANS': N_CHANS, 'N_CLUSTS': N_CLUSTS,
            'n_signals': n_signals, 'L': L, 'batch_norm': batch_norm,
            'up_method': up_method, 'last_act_fun': last_act_fun, 'G_params': G_params,
            'mse_est': error, 'n_params': n_params}
    timestamp = datetime.datetime.now().strftime("%Y_%m_%d-%H_%M")
    path = './results/test_input/input_noise_{}'.format(timestamp)
    np.save(path, data)
    print('SAVED as:',path)

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
    
    # Set seeds
    utils.RandomGraphSignal.set_seed(SEED)
    GraphDeepDecoder.set_seed(SEED)

    G = utils.create_graph(G_params)
    sizes, descendances, hier_As = compute_clusters(alg, G_params['k'])
    
    start_time = time.time()
    error = np.zeros((n_signals, N_SCENARIOS))
    with Pool() as pool:
        for i in range(n_signals):
            signal = utils.DeterministicGS(G,np.random.randn(N))
            signal.signal_to_0_1_interval()
            #signal.to_unit_norm()
            result = pool.apply_async(test_architecture,
                                        args=[i, signal.x, sizes,
                                                descendances, hier_As])

        for i in range(n_signals):
            error[i,:], n_params = result.get()

    # Print result:
    print('--- {} minutes ---'.format((time.time()-start_time)/60))
    print_results(N, np.mean(error, axis=0), n_params, np.median(error, axis=0))
    save_results(error, n_params, G_params)
    
    
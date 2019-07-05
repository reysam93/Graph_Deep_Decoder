"""
Check the efect of changing the size of the architecture for controlig the number of 
parameters which represent the signal
"""

import sys, os
import time, datetime
from multiprocessing import Pool, cpu_count 
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
n_p = 0.1 # SNR = 1/n_p
batch_norm = True #True
act_fun = nn.ReLU()
up_method = 'weighted'
last_act_fun = nn.Sigmoid() #nn.Tanh()

# Constants
SEED = 15
N_CHANS = [[6]*3, [4]*3, [3]*3, [2]*3, [10,5,3],  [4,3,3], [4], [9], [4]*4, [8,6,4,2],
           [4,3,2,2], [3]*6, [2]*6, [4, 4, 3, 3, 2]]
N_CLUSTS = [[4,16,64,256]]*6 + [[4,256]]*2 + [[4,16,32,64,256]]*3 + \
            [[4,8,16,32,64,128,256]]*2 + [[4, 8, 16 ,32, 64, 256]]

"""
N_CHANS = [[6]*3, [4]*3, [3]*3, [2]*3, [10,5,3],  [4,3,3], [4], [9], [4]*4, [8,6,4,2],
           [4,3,2,2], [3]*6, [2]*6, [4, 4, 3, 3, 2], [8, 8, 8, 8]]
N_CLUSTS = [[4,16,64,256]]*6 + [[4,256]]*2 + [[4,16,32,64,256]]*3 + \
            [[4,8,16,32,64,128,256]]*2 + [[4, 8, 16 ,32, 64, 256]] + \
            [[4, 16, 32, 64, 256]]
"""
N_SCENARIOS = len(N_CHANS)

def plot_clusters(G, cluster):
    G.set_coordinates(kind='community2D')
    _, axes = plt.subplots(1, 2)
    G.plot_signal(cluster.labels[0], ax=axes[0])
    axes[1].spy(G.W)
    plt.show()

def compute_clusters(G, alg, k):
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
    x_n = utils.GraphSignal.add_noise(x, n_p)
    for i in range(N_SCENARIOS):
        dec = GraphDeepDecoder(descendances[i], hier_As[i], sizes[i],
                        n_channels=N_CHANS[i], upsampling=up_method, batch_norm=batch_norm,
                        last_act_fun=last_act_fun, act_fun=act_fun)

        dec.build_network()
        x_est, mse_fit[i] = dec.fit(x_n)
        
        error[i] = np.sum(np.square(x-x_est))/np.square(np.linalg.norm(x))
        params[i] = dec.count_params()
        print('Signal: {} Scenario {}: ({} params): Error: {:.4f}'
                            .format(id, i+1, params[i], error[i]))
    return error, params

def print_results(N, err, params):
    mean_err = np.mean(err,0)
    median_err = np.median(err,0)
    std = np.std(error)
    for i in range(N_SCENARIOS):
        print('{}. (CHANS {}, CLUSTS: {}) '.format(i+1, N_CHANS[i], N_CLUSTS[i]))
        print('\tMean MSE: {}\tParams: {}\tCompression: {}\tMedian MSE: {}\tSTD: {}'
                            .format(mean_err[i], params[i], N/params[i], median_err[i], std[i]))

def save_results(error, n_params, G_params, n_p):
    if not os.path.isdir('./results/test_arch'):
        os.makedirs('./results/test_arch')

    data = {'SEED': SEED, 'N_CHANS': N_CHANS, 'N_CLUSTS': N_CLUSTS,
            'n_signals': n_signals, 'L': L, 'n_p': n_p, 'batch_norm': batch_norm,
            'up_method': up_method, 'last_act_fun': last_act_fun, 'G_params': G_params,
            'mse_est': error, 'n_params': n_params}
    timestamp = datetime.datetime.now().strftime("%Y_%m_%d-%H_%M")
    path = './results/test_arch/arch_pn_{}_{}'.format(n_p, timestamp)
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
    utils.GraphSignal.set_seed(SEED)
    GraphDeepDecoder.set_seed(SEED)

    G = utils.create_graph(G_params, SEED)
    sizes, descendances, hier_As = compute_clusters(G, alg, G_params['k'])

    start_time = time.time()
    error = np.zeros((n_signals, N_SCENARIOS))
    results = []
    with Pool(processes=cpu_count()) as pool:
        for i in range(n_signals):
            signal = utils.DifussedSparseGS(G,L,G_params['k'])
            signal.signal_to_0_1_interval()
            signal.to_unit_norm()
            results.append(pool.apply_async(test_architecture,
                                        args=[i, signal.x, sizes,
                                                descendances, hier_As]))

        for i in range(n_signals):
            error[i,:], n_params = results[i].get()

    # Print result:
    print('--- {} minutes ---'.format((time.time()-start_time)/60))
    print_results(N, error, n_params)
    save_results(error, n_params, G_params, n_p)
    
    
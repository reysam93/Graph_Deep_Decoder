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
n_signals = 3
L = 6
n_p = 0 # SNR = 1/n_p
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

def test_architecture(x, sizes, descendances, hier_As):
    error = np.zeros(N_SCENARIOS)
    mse_fit = np.zeros(N_SCENARIOS)
    params = np.zeros(N_SCENARIOS)
    x_n = utils.RandomGraphSignal.add_noise(x, n_p)
    for i in range(N_SCENARIOS):
        dec = GraphDeepDecoder(descendances[i], hier_As[i], sizes[i],
                        n_channels=N_CHANS[i], upsampling=up_method, batch_norm=batch_norm,
                        last_act_fun=last_act_fun, act_fun=act_fun)

        dec.build_network()
        x_est, mse_fit[i] = dec.fit(x_n)
        
        error[i] = np.sum(np.square(x-x_est))/np.linalg.norm(x)
        params[i] = dec.count_params()
        print('\tScenario {} ({} params): Error: {:.4f}'
                            .format(i, params[i], error[i]))
    mse_fit = mse_fit/np.linalg.norm(x_n)*x_n.size
    return error, params, mse_fit

def print_results(N, mean_error, params, mean_mse_fit):
    for i in range(N_SCENARIOS):
        print('{}. (CHANS {}, CLUSTS: {}) '.format(i+1, N_CHANS[i], N_CLUSTS[i]))
        print('\tMean MSE: {}\tParams: {}\tCompression: {}\tMSE fit {}'
                            .format(mean_error[i], params[i], N/params[i], mean_mse_fit[i]))

def save_results(error, mse_fit, n_params, G_params, n_p):
    if not os.path.isdir('./results/test_arch'):
        os.makedirs('./results/test_arch')

    data = {'SEED': SEED, 'N_CHANS': N_CHANS, 'N_CLUSTS': N_CLUSTS,
            'n_signals': n_signals, 'L': L, 'n_p': n_p, 'batch_norm': batch_norm,
            'up_method': up_method, 'last_act_fun': last_act_fun, 'G_params': G_params,
            'mse_est': error, 'mse_fit': mse_fit, 'n_params': n_params}
    timestamp = datetime.datetime.now().strftime("%Y_%m_%d-%H_%M")
    np.save('./results/test_arch/arch_pn_{}_{}'.format(n_p, timestamp), data)


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
    mse_fit = np.zeros((n_signals, N_SCENARIOS))
    error = np.zeros((n_signals, N_SCENARIOS))
    with Pool() as pool:
        for i in range(n_signals):
            signal = utils.DifussedSparseGS(G,L,G_params['k'])
            signal.signal_to_0_1_interval()
            signal.to_unit_norm()
            result = pool.apply_async(test_architecture,
                                        args=[signal.x, sizes, descendances, hier_As])

        signal.plot()

        for i in range(n_signals):
            print('Signal',i)
            error[i,:], n_params, mse_fit[i,:] = result.get()

    # Print result:
    print('--- {} minutes ---'.format((time.time()-start_time)/60))
    print_results(N, np.mean(error, axis=0), n_params, np.mean(mse_fit, axis=0))
    save_results(error, mse_fit, n_params, G_params, n_p)
    
    
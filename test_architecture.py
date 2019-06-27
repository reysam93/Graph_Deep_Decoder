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
import torch
import torch.nn as nn

# Tuning parameters
n_signals = 50
L = 5
n_p = 0.0 # SNR = 1/n_p
batch_norm = True #True
up_method = 'weighted'
last_act_fun = nn.Tanh() #nn.Sigmoid()

# Constants
SEED = 15
N_CHANS = [[4]*3, [6]*3, [2]*3, [2,2,1], [10,5,3], [4], [9], [4]*4, [3]*6, [8,6,4,2]]
N_CLUSTS = [[4,16,64,256]]*5 + [[4,256]]*2 + [[4,16,64,128,256]] + \
             [[4,8,16,32,64,128,256]] + [[4, 8, 16, 64, 256]]
N_SCENARIOS = len(N_CHANS)

def plot_clusters(G, cluster):
    G.set_coordinates(kind='community2D')
    _, axes = plt.subplots(1, 2)
    G.plot_signal(cluster.labels[0], ax=axes[0])
    axes[1].spy(G.W)
    plt.show()

def compute_clusters(alg):
    assert(len(N_CHANS) == len(N_CLUSTS))
    sizes = []
    descendances = []
    hier_As = []
    for i in range(len(N_CHANS)):
        assert(len(N_CHANS[i])+1 == len(N_CLUSTS[i]))
        cluster = utils.MultiRessGraphClustering(G, N_CLUSTS[i], alg)
        sizes.append(cluster.clusters_size)
        descendances.append(cluster.compute_hierarchy_descendance())
        hier_As.append(cluster.compute_hierarchy_A(up_method))
    return sizes, descendances, hier_As

def test_architecture(x, sizes, descendances, hier_As):
    mse_est = np.zeros(N_SCENARIOS)
    mse_fit = np.zeros(N_SCENARIOS)
    params = np.zeros(N_SCENARIOS)
    x_n = utils.DifussedSparseGraphSignal.add_noise(x, n_p)
    for i in range(N_SCENARIOS):
        dec = GraphDeepDecoder(descendances[i], hier_As[i], sizes[i],
                        n_channels=N_CHANS[i], upsampling=up_method, batch_norm=batch_norm,
                        last_act_fun=last_act_fun)

        dec.build_network()
        x_est, mse_fit[i] = dec.fit(x_n)
        
        mse_est[i] = np.mean(np.square(x-x_est))/np.linalg.norm(x)
        params[i] = dec.count_params()
    mse_fit = mse_fit/np.linalg.norm(x_n)
    return mse_est, params, mse_fit

def print_results(N, mean_mse, params, mean_mse_fit):
    for i in range(N_SCENARIOS):
        print('{}. (CHANS {}, CLUSTS: {}) '.format(i, N_CHANS[i], N_CLUSTS[i]))
        print('\tMean MSE: {}\tParams: {}\tCompression: {}\tMSE fit {}'
                            .format(mean_mse[i], params[i], N/params[i], mean_mse_fit[i]))

def save_results(mse_est, mse_fit, n_params, G_params):
    if not os.path.isdir('./results/test_arch'):
        os.makedirs('./results/test_arch')

    data = {'SEED': SEED, 'N_CHANS': N_CHANS, 'N_CLUSTS': N_CLUSTS,
            'n_signals': n_signals, 'L': L, 'n_p': n_p, 'batch_norm': batch_norm,
            'up_method': up_method, 'last_act_fun': last_act_fun, 'G_params': G_params,
            'mse_est': mse_est, 'mse_fit': mse_fit, 'n_params': n_params}
    timestamp = datetime.datetime.now().strftime("%Y_%m_%d-%H_%M")
    np.save('./results/arch_' + timestamp, data)


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
    utils.DifussedSparseGraphSignal.set_seed(SEED)
    GraphDeepDecoder.set_seed(SEED)

    G = utils.create_graph(G_params)
    sizes, descendances, hier_As = compute_clusters(alg)
    
    start_time = time.time()
    mse_fit = np.zeros((n_signals, N_SCENARIOS))
    mse_est = np.zeros((n_signals, N_SCENARIOS))
    with Pool() as pool:
        for i in range(n_signals):
            signal = utils.DifussedSparseGraphSignal(G,L,-1,1)
            signal.to_unit_norm()
            result = pool.apply_async(test_architecture,
                                        args=[signal.x, sizes, descendances, hier_As])

        for i in range(n_signals):
            mse_est[i,:], n_params, mse_fit[i,:] = result.get()
    
    # Print result:
    print('--- {} minutes ---'.format((time.time()-start_time)/60))
    print_results(N, np.mean(mse_est, axis=0), n_params, np.mean(mse_fit, axis=0))
    save_results(mse_est, mse_fit, n_params, G_params)
    
    
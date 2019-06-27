"""
Check the efect of the graph
"""

import sys, os
import time, datetime
from multiprocessing import Pool 
sys.path.insert(0, 'graph_deep_decoder')
from graph_deep_decoder import utils
from graph_deep_decoder.architecture import GraphDeepDecoder

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn


# Tuning parameters
n_signals = 50
L = 5
n_p = 0.1 # SNR = 1/n_p
batch_norm = True #True
up_method = 'weighted'
t = [4, 16, 64, None]
c_method = 'maxclust' # 'maxclust' or 'distance'
alg = 'spectral_clutering'
link = 'average'
n_chans = [2,2,2]
last_act_fun = nn.Tanh()
gamma = 0.5

# Constants
SEED = 15
G_PARAMS = [{'type': 'SBM','N': 256,'k': 2,'p': 0.1,'q': 0.01/4},
          {'type': 'SBM','N': 256,'k': 4,'p': 0.15,'q': 0.01/4},
          {'type': 'SBM','N': 256,'k': 10,'p': 0.4,'q': 0.01/4},
          {'type': 'SBM','N': 512,'k': 2,'p': 0.05,'q': 0.001},
          {'type': 'SBM','N': 512,'k': 4,'p': 0.075,'q': 0.001},
          {'type': 'SBM','N': 512,'k': 10,'p': 0.15,'q': 0.001},
          {'type': 'SBM','N': 1024,'k': 2,'p': 0.05,'q': 0.00075},
          {'type': 'SBM','N': 1024,'k': 4,'p': 0.075,'q': 0.0005},
          {'type': 'SBM','N': 1024,'k': 10,'p': 0.1,'q': 0.0005},
          {'type': 'ER','N': 256,'k': 4,'p': 0.05},
          {'type': 'ER','N': 512,'k': 4,'p': 0.025},
          {'type': 'ER','N': 1024,'k': 4,'p': 0.01}]
 
N_SCENARIOS = len(G_PARAMS)

def compute_clusters(Gs):
    sizes = []
    descendances = []
    hier_As = []

    for i, G in enumerate(Gs):
        t[-1] = G.N
        cluster = utils.MultiRessGraphClustering(G, t, G_PARAMS[i]['k'], alg,
                                                    method=c_method, link_fun=link)
        descendances.append(cluster.compute_hierarchy_descendance())
        hier_As.append(cluster.compute_hierarchy_A(up_method))
        sizes.append(cluster.clusters_size)
    return sizes, descendances, hier_As

def compute_signals(Gs):
    signals = []
    for i, G in enumerate(Gs):
        signal = utils.DifussedSparseGraphSignal(G,L,G_PARAMS[i]['k'],-1,1)
        signal.to_unit_norm()
        signals.append(signal.x)
    return signals

def test_graphs(signals, sizes, descendances, hier_As):
    mse_est = np.zeros(N_SCENARIOS)
    mse_fit = np.zeros(N_SCENARIOS)
    for i in range(N_SCENARIOS):
        x = signals[i]
        x_n = utils.DifussedSparseGraphSignal.add_noise(x, n_p)

        dec = GraphDeepDecoder(descendances[i], hier_As[i], sizes[i],
                        n_channels=n_chans, upsampling=up_method, batch_norm=batch_norm,
                        last_act_fun=last_act_fun, gamma=gamma)
        dec.build_network()
        x_est, mse_fit[i] = dec.fit(x_n)
        mse_est[i] = np.mean(np.square(x-x_est))/np.linalg.norm(x)
    mse_fit = mse_fit/np.linalg.norm(x_n)
    return mse_est, mse_fit

def print_results(mean_mse, mean_mse_fit, clust_sizes):
    for i in range(N_SCENARIOS):
        print('{}. (GRAPH: {}) '.format(i, G_PARAMS[i]))
        print('\tMean MSE: {}\tClust Sizes: {}\tMSE fit {}'
                            .format(mean_mse[i], clust_sizes[i], mean_mse_fit[i]))

def save_results(mse_est, mse_fit, n_params, G_params):
    if not os.path.isdir('./results/test_graph'):
        os.makedirs('./results/test_graph')

    data = {'SEED': SEED, 'G_PARAMS': G_PARAMS, 't': t, 'c_method': c_method, 'alg': alg,
            'n_signals': n_signals, 'L': L, 'n_p': n_p, 'batch_norm': batch_norm,
            'up_method': up_method, 'last_act_fun': last_act_fun, 'link': link,
            'n_chans': n_chans, 'gamma': gamma, 'mse_est': mse_est, 'mse_fit': mse_fit}
    
    timestamp = datetime.datetime.now().strftime("%Y_%m_%d-%H_%M")
    np.save('./results/test_graph/graph_' + timestamp, data)


if __name__ == '__main__':
    input_size = 4

    # Set seeds
    utils.DifussedSparseGraphSignal.set_seed(SEED)
    GraphDeepDecoder.set_seed(SEED)

    Gs = [utils.create_graph(G_PARAMS[i], seed=SEED) for i in range(N_SCENARIOS)] 
    sizes, descendances, hier_As = compute_clusters(Gs)
    
    start_time = time.time()
    mse_fit = np.zeros((n_signals, N_SCENARIOS))
    mse_est = np.zeros((n_signals, N_SCENARIOS))
    with Pool() as pool:
        for i in range(n_signals):
            signals = compute_signals(Gs)
            result = pool.apply_async(test_graphs, 
                                        args=[signals, sizes, descendances, hier_As])

        for i in range(n_signals):
            mse_est[i,:], mse_fit[i,:] = result.get()

    # Print result:
    print('--- {} minutes ---'.format((time.time()-start_time)/60))
    print_results(np.mean(mse_est, axis=0), np.mean(mse_fit, axis=0), sizes)
    save_results(mse_est, mse_fit, n_params, G_params)

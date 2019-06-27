"""
Check the efect of changing the resolution levels and the type of method used for 
defining the clusters after computing the distances
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
n_p = 0.1 # SNR = 1/n_p
alg = 'spectral_clutering'
batch_norm = True #True
up_method = 'weighted'
gamma = 0.5
n_chans = [2,2,2]
last_act_fun = nn.Tanh()

# Constants
SEED = 15
RESOLUTIONS = [['maxclust', [4, 8, 16, 256], 4],
                ['maxclust', [4, 16, 64, 256], 4],
                ['maxclust', [4, 32, 128, 256], 4],
                ['maxclust', [2, 8, 32, 256], 2],
                ['maxclust', [2, 4, 64, 256], 2],
                ['distance',  [1, .75, .5, 0], 4],
                ['distance',  [1, .66, .33, 0], 4],
                ['distance',  [1, .8, .4, 0], 4],
                ['distance',  [1, .75, .5, 0], 2],
                ['distance',  [1, .3, .15, 0], 2]]
N_SCENARIOS = len(RESOLUTIONS)

# NOTE: compute_clusters is always the same..., only changing the arguments it uses (?)
def compute_clusters(k):
    sizes = []
    descendances = []
    hier_As = []
    for i in range(N_SCENARIOS):
        cluster = utils.MultiRessGraphClustering(G, RESOLUTIONS[i][1], RESOLUTIONS[i][2],
                                        alg, method=RESOLUTIONS[i][0])
        sizes.append(cluster.clusters_size)
        descendances.append(cluster.compute_hierarchy_descendance())
        hier_As.append(cluster.compute_hierarchy_A(up_method))

    return sizes, descendances, hier_As

def test_resolution(x, sizes, descendances, hier_As):
    mse_est = np.zeros(N_SCENARIOS)
    mse_fit = np.zeros(N_SCENARIOS)
    x_n = utils.DifussedSparseGraphSignal.add_noise(x, n_p)
    for i in range(N_SCENARIOS):
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
        print('{}. (RES: {}) '.format(i, RESOLUTIONS[i]))
        print('\tMean MSE: {}\tClust Sizes: {}\tMSE fit {}'
                            .format(mean_mse[i], clust_sizes[i], mean_mse_fit[i]))

def save_results(mse_est, mse_fit, G_params):
    if not os.path.isdir('./results/test_res'):
        os.makedirs('./results/test_arch')

    data = {'SEED': SEED, 'RESOLUTIONS': RESOLUTIONS, 'alg': alg, 'gamma': gamma,
            'n_signals': n_signals, 'L': L, 'n_p': n_p, 'batch_norm': batch_norm,
            'up_method': up_method, 'last_act_fun': last_act_fun, 'G_params': G_params,
            'mse_est': mse_est, 'mse_fit': mse_fit, 'n_chans': n_chans}
    timestamp = datetime.datetime.now().strftime("%Y_%m_%d-%H_%M")
    np.save('./results/res_' + timestamp, data)

if __name__ == '__main__':
    # Graph parameters
    G_params = {}
    G_params['type'] = 'SBM' # SBM or ER
    G_params['N'] = N = 256
    G_params['k'] = 4
    G_params['p'] = 0.15
    G_params['q'] = 0.01/4

    # Set seeds
    utils.DifussedSparseGraphSignal.set_seed(SEED)
    GraphDeepDecoder.set_seed(SEED)

    G = utils.create_graph(G_params)    
    sizes, descendances, hier_As = compute_clusters(G_params['k'])

    start_time = time.time()
    mse_fit = np.zeros((n_signals, N_SCENARIOS))
    mse_est = np.zeros((n_signals, N_SCENARIOS))
    with Pool() as pool:
        for i in range(n_signals):
            signal = utils.DifussedSparseGraphSignal(G,L,G_params['k'])
            signal.to_unit_norm()
            result = pool.apply_async(test_resolution,
                                        args=[signal.x, sizes, descendances, hier_As])

        for i in range(n_signals):
            mse_est[i,:], mse_fit[i,:] = result.get()

    # Print result:
    print('--- {} minutes ---'.format((time.time()-start_time)/60))
    print_results(np.mean(mse_est, axis=0), np.mean(mse_fit, axis=0), sizes)
    save_results(mse_est, mse_fit, G_params)
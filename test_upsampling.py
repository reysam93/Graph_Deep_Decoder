"""
Check the efect of changing the upsampling algorithm for reconstructing the graph 
signal from noise.
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
n_signals = 200
L = 6
n_p = 0 # SNR = 1/n_p
type_z = 'alternated'
batch_norm = True
t = [4, 16, 64, 256] # Max clusters
c_method = 'maxclust' # 'maxclust' or 'distance'
alg = 'spectral_clutering'
linkage = 'average'
n_chans = [3,3,3]
last_act_fun = nn.Sigmoid()


# Constants
SEED = 15
UPSAMPLING = [[None, None], ['original', None], ['no_A', None],
              ['binary', 0], ['binary', .25], ['binary', .5], ['binary', .75],
              ['weighted', 0], ['weighted', .25], ['weighted', .5], ['weighted', .75]]
N_SCENARIOS = len(UPSAMPLING)

def compute_clusters(k):
    sizes = []
    descendances = []
    hier_As = []
    for i in range(N_SCENARIOS):
        cluster = utils.MultiRessGraphClustering(G, t, k, alg, method=c_method,
                                                        link_fun=linkage)
        sizes.append(cluster.clusters_size)
        descendances.append(cluster.compute_hierarchy_descendance())
        hier_As.append(cluster.compute_hierarchy_A(UPSAMPLING[i][0]))

    return sizes, descendances, hier_As

def test_upsampling(x, sizes, descendances, hier_As):
    mse_est = np.zeros(N_SCENARIOS)
    mse_fit = np.zeros(N_SCENARIOS)
    x_n = utils.RandomGraphSignal.add_noise(x, n_p)
    for i in range(N_SCENARIOS):
        dec = GraphDeepDecoder(descendances[i], hier_As[i], sizes[i],
                        n_channels=n_chans, upsampling=UPSAMPLING[i][0], 
                        batch_norm=batch_norm, last_act_fun=last_act_fun,
                        gamma=UPSAMPLING[i][1])

        dec.build_network()
        x_est, mse_fit[i] = dec.fit(x_n)
        
        mse_est[i] = np.mean(np.square(x-x_est))/np.linalg.norm(x)
    mse_fit = mse_fit/np.linalg.norm(x_n)
    return mse_est, mse_fit

def print_results(mean_mse, mean_mse_fit):
    for i in range(N_SCENARIOS):
        print('{}. (UPSAMPLING: {}) '.format(i, UPSAMPLING[i]))
        print('\tMean MSE: {}\tMSE fit {}'.format(mean_mse[i], mean_mse_fit[i]))

def save_results(mse_est, mse_fit, G_params, n_p):
    if not os.path.isdir('./results/test_ups'):
        os.makedirs('./results/test_ups')

    data = {'SEED': SEED, 'UPSAMPLING': UPSAMPLING, 'type_z': type_z, 't': t,
            'n_signals': n_signals, 'L': L, 'n_p': n_p, 'batch_norm': batch_norm,
            'c_method': c_method, 'alg': alg, 'last_act_fun': last_act_fun,
            'G_params': G_params, 'linkage': linkage, 'n_chans': n_chans,
            'mse_est': mse_est, 'mse_fit': mse_fit}
    timestamp = datetime.datetime.now().strftime("%Y_%m_%d-%H_%M")
    path = './results/test_ups/ups_pn_{}_{}'.format(n_p, timestamp)
    np.save(path, data)
    print('SAVED as:', path)

if __name__ == '__main__':
    # Graph parameters
    G_params = {}
    G_params['type'] = 'SBM' # SBM or ER
    G_params['N'] = N = 256
    G_params['k'] = 4
    G_params['p'] = 0.15
    G_params['q'] = 0.01/4

    # Set seeds
    utils.RandomGraphSignal.set_seed(SEED)
    GraphDeepDecoder.set_seed(SEED)

    G = utils.create_graph(G_params, SEED, type_z)   
    sizes, descendances, hier_As = compute_clusters(G_params['k'])
    
    start_time = time.time()
    mse_fit = np.zeros((n_signals, N_SCENARIOS))
    mse_est = np.zeros((n_signals, N_SCENARIOS))
    with Pool() as pool:
        for i in range(n_signals):
            signal = utils.DifussedSparseGS(G,L,G_params['k'])
            signal.to_unit_norm()
            
            result = pool.apply_async(test_upsampling,
                                        args=[signal.x, sizes, descendances, hier_As])

        for i in range(n_signals):
            mse_est[i,:], mse_fit[i,:] = result.get()

    # Print result:
    print('--- {} minutes ---'.format((time.time()-start_time)/60))
    print_results(np.mean(mse_est, axis=0), np.mean(mse_fit, axis=0))
    save_results(mse_est, mse_fit, G_params, n_p)
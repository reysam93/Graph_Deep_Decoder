"""
Check the efect of changing the clustering algorithm used for constructing the
upsampling matrix
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
L = 5
n_p = 0 # SNR = 1/n_p
batch_norm = True #True
up_method = 'weighted'
t = [4, 16, 64, 256] # Max clusters
#t = [1, 0.75, 0.5, 0] # relative max distances
c_method = 'maxclust' # 'maxclust' or 'distance'
n_chans = [3,3,3]
last_act_fun = nn.Sigmoid()

# Constants
SEED = 15
CLUST_ALGS = [['spectral_clutering', 'single'],
              ['spectral_clutering', 'complete'],
              ['spectral_clutering', 'average'],
              ['spectral_clutering', 'ward'],
              #['distance_clustering', 'single'],
              ['distance_clustering', 'complete'],
              ['distance_clustering', 'average'],
              ['distance_clustering', 'ward']]
N_SCENARIOS = len(CLUST_ALGS)


def compute_clusters(k):
    sizes = []
    descendances = []
    hier_As = []
    max_dists = []
    for i in range(N_SCENARIOS):
        alg = CLUST_ALGS[i][0]
        link = CLUST_ALGS[i][1]
        cluster = utils.MultiRessGraphClustering(G, t, k, alg, method=c_method,
                                                        link_fun=link)
        sizes.append(cluster.clusters_size)
        descendances.append(cluster.compute_hierarchy_descendance())
        hier_As.append(cluster.compute_hierarchy_A(up_method))
        max_dists.append(cluster.Z[:,2][-k]/cluster.Z[:,2][-1])
    return sizes, descendances, hier_As, max_dists

def test_clustering(id, x, sizes, descendances, hier_As):
    mse_est = np.zeros(N_SCENARIOS)
    x_n = utils.GraphSignal.add_noise(x, n_p)

    for i in range(N_SCENARIOS):
        dec = GraphDeepDecoder(descendances[i], hier_As[i], sizes[i],
                        n_channels=n_chans, upsampling=up_method, batch_norm=batch_norm,
                        last_act_fun=last_act_fun)

        dec.build_network()
        x_est, _ = dec.fit(x_n)
        mse_est[i] = np.sum(np.square(x-x_est))/np.square(np.linalg.norm(x))
        print('Signal: {} Exp: {}: Err: {}'.format(id, i, mse_est[i]))
    return mse_est

def print_results(mean_mse, median_mse, clust_sizes, max_dists):
    for i in range(N_SCENARIOS):
        print('{}. (ALG: {}, LINK: {}) '.format(i, CLUST_ALGS[i][0], CLUST_ALGS[i][1]))
        print('\tMean MSE: {}\tClust Sizes: {}\tMax Dist: {}\tMedian MSE: {}'
                            .format(mean_mse[i], clust_sizes[i], max_dists[i], median_mse[i]))

def save_results(mse_est, n_p, G_params):
    if not os.path.isdir('./results/test_clust'):
        os.makedirs('./results/test_clust')

    data = {'SEED': SEED, 'CLUST_ALGS': CLUST_ALGS, 'n_signals': n_signals, 'L': L, 
            'n_p': n_p, 'batch_norm': batch_norm, 'up_method': up_method, 't': t,
            'c_method': c_method, 'n_chans': n_chans, 'last_act_fun': last_act_fun,
            'G_params': G_params, 'mse_est': mse_est}
    timestamp = datetime.datetime.now().strftime("%Y_%m_%d-%H_%M")
    path = './results/test_clust/clust_p_n_{}_{}'.format(n_p, timestamp)
    np.save(path, data)
    print('SAVED as:', path)


# Tuning parameters
t = [4, 16, 64, 256] # Max clusters
#t = [1, 0.75, 0.5, 0] # relative max distances
c_method = 'maxclust' # 'maxclust' or 'distance'
n_chans = [2,2,2]
last_act_fun = nn.Tanh()


if __name__ == '__main__':
    # Graph parameters
    G_params = {}
    G_params['type'] = 'SBM' # SBM or ER
    G_params['N'] = N = 256
    G_params['k'] = 4
    G_params['p'] = 0.15
    G_params['q'] = 0.01/4

    # Set seeds
    utils.GraphSignal.set_seed(SEED)
    GraphDeepDecoder.set_seed(SEED)

    G = utils.create_graph(G_params)  
    sizes, descendances, hier_As, max_dists = compute_clusters(G_params['k'])

    start_time = time.time()
    mse_est = np.zeros((n_signals, N_SCENARIOS))
    with Pool() as pool:
        for i in range(n_signals):
            signal = utils.DifussedSparseGS(G,L,G_params['k'])
            signal.signal_to_0_1_interval()
            signal.to_unit_norm()
            result = pool.apply_async(test_clustering,
                                        args=[i, signal.x, sizes, descendances, hier_As])

        for i in range(n_signals):
            mse_est[i,:] = result.get()
    
    # Print result:
    print('--- {} minutes ---'.format((time.time()-start_time)/60))
    print_results(np.mean(mse_est, axis=0), np.median(mse_est, axis=0), sizes, max_dists)
    save_results(mse_est, n_p, G_params)

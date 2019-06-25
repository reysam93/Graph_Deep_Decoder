"""
Check the efect of changing the upsampling algorithm for reconstructing the graph 
signal from noise.
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
import torch
import torch.nn as nn

# Tuning parameters
n_signals = 50
L = 5
n_p = 0.0 # SNR = 1/n_p
batch_norm = True
t = [4, 16, 64, 256] # Max clusters
c_method = 'maxclust' # 'maxclust' or 'distance'
alg = 'spectral_clutering'
linkage = 'average'
n_chans = [2,2,2]
last_act_fun = nn.Tanh()


# Constants
SEED = 15
UPSAMPLING = [[None, None], ['original', None], ['no_A', None],
              ['binary', 0], ['binary', .25], ['binary', .5], ['binary', .75],
              ['weighted', 0], ['weighted', .25], ['weighted', .5], ['weighted', .75]]
N_SCENARIOS = len(UPSAMPLING)

def compute_clusters():
    sizes = []
    descendances = []
    hier_As = []
    for i in range(N_SCENARIOS):
        cluster = utils.MultiRessGraphClustering(G, t, alg, method=c_method,
                                                        link_fun=linkage)
        sizes.append(cluster.clusters_size)
        descendances.append(cluster.compute_hierarchy_descendance())
        hier_As.append(cluster.compute_hierarchy_A(UPSAMPLING[i][0]))

    return sizes, descendances, hier_As

def test_upsampling(x, sizes, descendances, hier_As):
    mse_est = np.zeros(N_SCENARIOS)
    mse_fit = np.zeros(N_SCENARIOS)
    x_n = x + np.random.randn(x.size)*np.sqrt(n_p)
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


if __name__ == '__main__':
    # Graph parameters
    N = 256
    k = 4
    p = 0.15 #1/math.log(10*N)
    q = 0.01/(k)

    # Set seeds
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    z = np.array(list(range(k))*int(N/k)+list(range(N%k)))
    #z = None
    G = StochasticBlockModel(N=N, k=k, p=p, q=q, z=z, connected=True, seed=SEED)    
    sizes, descendances, hier_As = compute_clusters()
    
    start_time = time.time()
    mse_fit = np.zeros((n_signals, N_SCENARIOS))
    mse_est = np.zeros((n_signals, N_SCENARIOS))
    with Pool() as pool:
        for i in range(n_signals):
            signal = utils.DifussedSparseGraphSignal(G,L,-1,1)
            signal.to_unit_norm()
            
            
            result = pool.apply_async(test_upsampling,
                                        args=[signal.x, sizes, descendances, hier_As])

        for i in range(n_signals):
            mse_est[i,:], mse_fit[i,:] = result.get()

    # Print result:
    print('--- {} minutes ---'.format((time.time()-start_time)/60))
    print_results(np.mean(mse_est, axis=0), np.mean(mse_fit, axis=0))

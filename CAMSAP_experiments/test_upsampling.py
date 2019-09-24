"""
Check the efect of changing the upsampling algorithm for reconstructing the graph 
signal from noise.
"""

import sys, os
import time, datetime
from multiprocessing import Pool, cpu_count
sys.path.insert(0, '../graph_deep_decoder')
from graph_deep_decoder import utils
from graph_deep_decoder import graph_signals as gs
from graph_deep_decoder.architecture import GraphDeepDecoder
from pygsp.graphs import StochasticBlockModel, ErdosRenyi

import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn

# Tuning parameters
n_signals = 100
L = 6
type_z = 'random'
batch_norm = True
t = [4, 16, 64, 256] 
c_method = 'maxclust'
alg = 'spectral_clutering'
linkage = 'average'
n_chans = [3,3,3]
last_act_fun = nn.Sigmoid()


# Constants
N_CPUS = cpu_count()-1
SAVE = False
SEED = 15
N_P = [0, .05, .1, .15, .2, .25, .3, .35, .4, .45, .5]

EXPERIMENTS = [[None, None], ['original', None], ['no_A', None],
              ['binary', .5], ['weighted', 0], ['weighted', .25],
              ['weighted', .5], ['weighted', .75]]

"""
EXPERIMENTS = [[None, None], ['original', None], ['no_A', None],
              ['binary', 0], ['binary', .25], ['binary', .5], ['binary', .75],
              ['weighted', 0], ['weighted', .25], ['weighted', .5], ['weighted', .75]]
"""
N_SCENARIOS = len(EXPERIMENTS)

def compute_clusters(k):
    sizes = []
    descendances = []
    hier_As = []
    for i in range(N_SCENARIOS):
        cluster = utils.MultiRessGraphClustering(G, t, k, alg, method=c_method,
                                                        link_fun=linkage)
        sizes.append(cluster.clusters_size)
        descendances.append(cluster.compute_hierarchy_descendance())
        hier_As.append(cluster.compute_hierarchy_A(EXPERIMENTS[i][0]))

    return sizes, descendances, hier_As

def test_upsampling(id, x, sizes, descendances, hier_As, n_p):
    error = np.zeros(N_SCENARIOS)
    mse_fit = np.zeros(N_SCENARIOS)
    x_n = gs.GraphSignal.add_noise(x, n_p)
    for i in range(N_SCENARIOS):
        dec = GraphDeepDecoder(descendances[i], hier_As[i], sizes[i],
                        n_channels=n_chans, upsampling=EXPERIMENTS[i][0], 
                        batch_norm=batch_norm, last_act_fun=last_act_fun,
                        gamma=EXPERIMENTS[i][1])

        dec.build_network()
        x_est, mse_fit[i] = dec.fit(x_n)

        error[i] = np.sum(np.square(x-x_est))/np.square(np.linalg.norm(x))
        print('Signal: {} Scenario {}: Error: {:.4f}'
                            .format(id, i+1, error[i]))
    mse_fit = mse_fit/np.linalg.norm(x_n)*x_n.size
    return error

def print_results(mse_est):
    mean_mse = np.mean(mse_est, axis=0)
    median_mse = np.median(mse_est, axis=0)
    std = np.std(mse_est, axis=0)
    for i in range(N_SCENARIOS):
        print('{}. {} '.format(i+1, EXPERIMENTS[i]))
        print('\tMean MSE: {}\tMedian MSE: {}\tSTD: {}'.format(mean_mse[i], median_mse[i], std[i]))

def save_partial_results(error, G_params, n_p):
    if not os.path.isdir('./results/test_ups'):
        os.makedirs('./results/test_ups')

    data = {'SEED': SEED, 'EXPERIMENTS': EXPERIMENTS, 'type_z': type_z, 't': t,
            'n_signals': n_signals, 'L': L, 'n_p': n_p, 'batch_norm': batch_norm,
            'c_method': c_method, 'alg': alg, 'last_act_fun': last_act_fun,
            'G_params': G_params, 'linkage': linkage, 'n_chans': n_chans,
            'error': error}
    timestamp = datetime.datetime.now().strftime("%Y_%m_%d-%H_%M")
    path = './results/test_ups/ups_pn_{}_{}'.format(n_p, timestamp)
    np.save(path, data)
    print('SAVED as:', path)

def save_results(error, G_params):
    if not os.path.isdir('./results/test_ups'):
        os.makedirs('./results/test_ups')

    data = {'SEED': SEED, 'EXPERIMENTS': EXPERIMENTS, 'type_z': type_z, 't': t,
            'n_signals': n_signals, 'L': L, 'N_P': N_P, 'batch_norm': batch_norm,
            'c_method': c_method, 'alg': alg, 'last_act_fun': last_act_fun,
            'G_params': G_params, 'linkage': linkage, 'n_chans': n_chans,
            'error': error}
    timestamp = datetime.datetime.now().strftime("%Y_%m_%d-%H_%M")
    path = './results/test_ups/ups_{}'.format(timestamp)
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
    gs.GraphSignal.set_seed(SEED)
    GraphDeepDecoder.set_seed(SEED)

    G = utils.create_graph(G_params, SEED, type_z)   
    sizes, descendances, hier_As = compute_clusters(G_params['k'])
    
    print("CPUs used:", N_CPUS)
    start_time = time.time()
    error = np.zeros((len(N_P), n_signals, N_SCENARIOS))
    for i, n_p in enumerate(N_P):
        print('Noise:', n_p)
        results = []
        with Pool(processes=N_CPUS) as pool:
            for j in range(n_signals):
                signal = gs.DifussedSparseGS(G,L,G_params['k'])            
                signal.signal_to_0_1_interval()
                signal.to_unit_norm()
                
                results.append(pool.apply_async(test_upsampling,
                                            args=[j, signal.x, sizes,
                                                    descendances, hier_As, n_p]))

            for j in range(n_signals):
                error[i,j,:] = results[j].get()

        # Print result:
        print_results(error[i,:,:])
    print('--- {} hours ---'.format((time.time()-start_time)/3600))
    if SAVE:
        save_results(error, G_params)

"""
Check the efect of changing the size of the architecture for controlig the number of 
parameters which represent the signal
"""

import sys, os
import time, datetime
from multiprocessing import Pool, cpu_count 
sys.path.insert(0, 'graph_deep_decoder')
from graph_deep_decoder import utils
from graph_deep_decoder import graph_signals as gs
from graph_deep_decoder.architecture import GraphDeepDecoder
from pygsp.graphs import StochasticBlockModel, ErdosRenyi

import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn

# Tuning parameters
n_signals = 50
L = 6
batch_norm = True
act_fun = nn.Tanh()
up_method = 'weighted'
gamma = 0.5
type_z = 'alternated'
last_act_fun = None

# Constants
SAVE = True
SEED = 15
N_P = [0, .2]
EXPERIMENTS = [{'n_chans': [6]*3, 'n_clusts': [4,16,64,256]},
               {'n_chans': [4]*4, 'n_clusts': [4,16,32,64,256]},
               {'n_chans': [4,4,3,3,2], 'n_clusts': [4,8,16,32,64,256]},
               {'n_chans': [3]*3, 'n_clusts': [4,16,64,256]},
               {'n_chans': [2]*6, 'n_clusts': [4,8,16,32,64,128,256]}]

N_EXPS = len(EXPERIMENTS)

def plot_clusters(G, cluster):
    G.set_coordinates(kind='community2D')
    _, axes = plt.subplots(1, 2)
    G.plot_signal(cluster.labels[0], ax=axes[0])
    axes[1].spy(G.W)
    plt.show()

def compute_clusters(G, alg, k):
    sizes = []
    descendances = []
    hier_As = []
    for i in range(N_EXPS):
        cluster = utils.MultiRessGraphClustering(G, EXPERIMENTS[i]['n_clusts'], k, alg)
        sizes.append(cluster.clusters_size)
        descendances.append(cluster.compute_hierarchy_descendance())
        hier_As.append(cluster.compute_hierarchy_A(up_method))
    return sizes, descendances, hier_As

def test_architecture(id, x, sizes, descendances, hier_As, n_p):    
    error = np.zeros(N_EXPS)
    mse_fit = np.zeros(N_EXPS)
    params = np.zeros(N_EXPS)
    x_n = gs.GraphSignal.add_noise(x, n_p)
    for i in range(N_EXPS):
        dec = GraphDeepDecoder(descendances[i], hier_As[i], sizes[i],
                        n_channels=EXPERIMENTS[i]['n_chans'], upsampling=up_method, batch_norm=batch_norm,
                        last_act_fun=last_act_fun, act_fun=act_fun, gamma=gamma)

        dec.build_network()
        x_est, mse_fit[i] = dec.fit(x_n, n_iter=3000)
        
        error[i] = np.sum(np.square(x-x_est))/np.square(np.linalg.norm(x))
        params[i] = dec.count_params()
        print('Signal: {} Scenario {}: ({} params): Error: {:.4f}'
                            .format(id, i+1, params[i], error[i]))
    return error, params

def print_results(N, err, params):
    mean_err = np.mean(err,0)
    median_err = np.median(err,0)
    std = np.std(err,0)
    for i in range(N_EXPS):
        print('{}. {} '.format(i+1, EXPERIMENTS[i]))
        print('\tMean MSE: {}\tParams: {}\tCompression: {}\tMedian MSE: {}\tSTD: {}'
                            .format(mean_err[i], params[i], N/params[i], median_err[i], std[i]))

def save_partial_results(error, n_params, G_params, n_p):
    if not os.path.isdir('./results/test_arch'):
        os.makedirs('./results/test_arch')

    data = {'SEED': SEED, 'EXPERIMENTS': EXPERIMENTS,
            'n_signals': n_signals, 'L': L, 'n_p': n_p, 'batch_norm': batch_norm,
            'up_method': up_method, 'last_act_fun': last_act_fun, 'G_params': G_params,
            'mse_est': error, 'n_params': n_params}
    timestamp = datetime.datetime.now().strftime("%Y_%m_%d-%H_%M")
    path = './results/test_arch/arch_pn_{}_{}'.format(n_p, timestamp)
    np.save(path, data)
    print('SAVED as:',path)

def save_results(error, n_params, G_params):
    if not os.path.isdir('./results/test_arch'):
        os.makedirs('./results/test_arch')

    data = {'SEED': SEED, 'EXPERIMENTS': EXPERIMENTS,
            'n_signals': n_signals, 'L': L, 'N_P': N_P, 'batch_norm': batch_norm,
            'up_method': up_method, 'last_act_fun': last_act_fun, 'G_params': G_params,
            'error': error, 'n_params': n_params, 'gamma': gamma, 'type_z': type_z}
    timestamp = datetime.datetime.now().strftime("%Y_%m_%d-%H_%M")
    path = './results/test_arch/arch_{}'.format(timestamp)
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
    gs.GraphSignal.set_seed(SEED)
    GraphDeepDecoder.set_seed(SEED)

    G = utils.create_graph(G_params, SEED, type_z=type_z)
    sizes, descendances, hier_As = compute_clusters(G, alg, G_params['k'])


    start_time = time.time()
    error = np.zeros((len(N_P), n_signals, N_EXPS))
    for i, n_p in enumerate(N_P):
        print('Noise:', n_p)
        results = []
        with Pool(processes=cpu_count()) as pool:
            for j in range(n_signals):
                signal = gs.DifussedSparseGS(G,L,G_params['k'])
                signal.to_unit_norm()
                results.append(pool.apply_async(test_architecture,
                                           args=[j, signal.x, sizes,
                                                descendances, hier_As, n_p]))
            for j in range(n_signals):
                error[i,j,:], n_params = results[j].get()

        # Print result:
        print_results(N, error[i,:,:], n_params)
    print('--- {} hours ---'.format((time.time()-start_time)/3600))
    if SAVE:
        save_results(error, n_params, G_params)
    
    
"""
Check the efect of changing the size of the architecture for controlig the
number of parameters which represent the signal
"""

import sys
import os
import time
import datetime
from multiprocessing import Pool, cpu_count

import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
from torch import manual_seed

sys.path.insert(0, '../graph_deep_decoder')
from graph_deep_decoder import datasets as ds
from graph_deep_decoder import graph_clustering as gc
from graph_deep_decoder.architecture import GraphDeepDecoder


# Tuning parameters
n_signals = 100
L = 6
batch_norm = True
act_fun = nn.ReLU()
up_method = gc.WEI
gamma = 0.5
last_act_fun = nn.Sigmoid()

# Constants
N_CPUS = cpu_count
SAVE = True
SEED = 15
N_P = [0, .1, .2, .3, .4, .5]
EXPS = [{'feats': [6,6,6,6,6,1], 'nodes': [4,16,64,256,256,256]},
        {'feats': [4,4,4,4,4,4,1], 'nodes': [4,16,32,64,256,256,256]},
        {'feats': [3,3,3,3,3,1], 'nodes': [4,16,64,256,256,256]},
        {'feats': [2,2,2,2,2,2,2,2,1], 'nodes': [4,8,16,32,64,128,256,256,256]}]

N_EXPS = len(EXPS)


def plot_clusters(G, cluster):
    G.set_coordinates(kind='community2D')
    _, axes = plt.subplots(1, 2)
    G.plot_signal(cluster.labels[0], ax=axes[0])
    axes[1].spy(G.W)
    plt.show()


def compute_clusters(G, alg, k):
    sizes = []
    Us = []
    hier_As = []
    for i in range(N_EXPS):
        cluster = gc.MultiResGraphClustering(G, EXPS[i]['nodes'], k, alg)
        sizes.append(cluster.sizes)
        Us.append(cluster.Us)
        hier_As.append(cluster.As)
    return sizes, Us, hier_As


def test_architecture(id, x, sizes, Us, As, n_p):
    error = np.zeros(N_EXPS)
    mse_fit = np.zeros(N_EXPS)
    params = np.zeros(N_EXPS)
    x_n = ds.GraphSignal.add_noise(x, n_p)
    for i in range(N_EXPS):
        dec = GraphDeepDecoder(EXPS[i]['feats'], sizes[i], Us[i], As=As[i],
                               ups=up_method, gamma=gamma,
                               batch_norm=batch_norm, act_fn=act_fun,
                               last_act_fn=last_act_fun)

        # Implement model??
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
        print('{}. {} '.format(i+1, EXPS[i]))
        print('\tMean MSE: {}\tParams: {}\tCompression: {}\tMedian MSE: {}\tSTD: {}'
              .format(mean_err[i], params[i], N/params[i], median_err[i], std[i]))


def save_partial_results(error, n_params, G_params, n_p):
    if not os.path.isdir('./results/test_arch'):
        os.makedirs('./results/test_arch')

    data = {'SEED': SEED, 'EXPERIMENTS': EXPS,
            'n_signals': n_signals, 'L': L, 'n_p': n_p, 'batch_norm': batch_norm,
            'up_method': up_method, 'last_act_fun': last_act_fun, 'G_params': G_params,
            'mse_est': error, 'n_params': n_params}
    timestamp = datetime.datetime.now().strftime("%Y_%m_%d-%H_%M")
    path = './results/test_arch/arch_pn_{}_{}'.format(n_p, timestamp)
    np.save(path, data)
    print('SAVED as:', path)


def save_results(error, n_params, G_params):
    if not os.path.isdir('./results/test_arch'):
        os.makedirs('./results/test_arch')

    data = {'SEED': SEED, 'EXPERIMENTS': EXPS,
            'n_signals': n_signals, 'L': L, 'N_P': N_P, 'batch_norm': batch_norm,
            'up_method': up_method, 'last_act_fun': last_act_fun, 'G_params': G_params,
            'error': error, 'n_params': n_params, 'gamma': gamma}
    timestamp = datetime.datetime.now().strftime("%Y_%m_%d-%H_%M")
    path = './results/test_arch/arch_{}'.format(timestamp)
    np.save(path, data)
    print('SAVED as:', path)


if __name__ == '__main__':
    # Graph parameters
    G_params = {}
    G_params['type'] = ds.SBM  # SBM or ER
    G_params['N'] = N = 256
    G_params['k'] = k = 4
    G_params['p'] = 0.15
    G_params['q'] = 0.01/k
    G_params['type_z'] = ds.RAND

    # Cluster Params
    alg = 'spectral_clutering'  # spectral_clutering or distance_clustering
    # method = 'maxclust'

    # Set seeds
    np.random.seed(SEED)
    manual_seed(SEED)

    G = ds.create_graph(G_params, SEED)
    sizes, Us, As = compute_clusters(G, alg, G_params['k'])

    print("CPUs used:", N_CPUS)
    start_time = time.time()
    error = np.zeros((len(N_P), n_signals, N_EXPS))
    for i, n_p in enumerate(N_P):
        print('Noise:', n_p)
        results = []
        with Pool(processes=N_CPUS) as pool:
            for j in range(n_signals):
                signal = ds.DifussedSparseGS(G, L, G_params['k'])
                signal.signal_to_0_1_interval()
                signal.to_unit_norm()
                results.append(pool.apply_async(test_architecture,
                               args=[j, signal.x, sizes, Us, As, n_p]))
            for j in range(n_signals):
                error[i, j, :], n_params = results[j].get()

        # Print result:
        print_results(N, error[i, :, :], n_params)
    print('--- {} hours ---'.format((time.time()-start_time)/3600))
    if SAVE:
        save_results(error, n_params, G_params)

"""
Check the efect of changing the upsampling algorithm for reconstructing the graph 
signal from noise.
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
t = [4, 16, 64, 256, 256, 256]
c_method = 'maxclust'
alg = 'spectral_clutering'
n_chans = [3, 3, 3, 3, 3, 1]
last_act_fun = nn.Sigmoid()
act_fun = nn.ReLU()

# Constants
N_CPUS = cpu_count()
SAVE = True
SEED = 10
N_P = [0, .1, .2, .3, .4, .5]

EXPS = [[None, None], [gc.REG, None], [gc.NO_A, None],
        [gc.BIN, .5], [gc.WEI, .5]]

N_SCENARIOS = len(EXPS)


def compute_clusters(k):
    sizes = []
    Us = []
    hier_As = []
    for i in range(N_SCENARIOS):
        if EXPS[i][0] is None:
            nodes = [t[-1]]*len(t)
        else:
            nodes = t
        cluster = gc.MultiResGraphClustering(G, nodes, k, alg, method=c_method,
                                             up_method=EXPS[i][0])
        sizes.append(cluster.sizes)
        Us.append(cluster.Us)
        hier_As.append(cluster.As)
    return sizes, Us, hier_As


def test_upsampling(id, x, sizes, Us, As, n_p):
    error = np.zeros(N_SCENARIOS)
    mse_fit = np.zeros(N_SCENARIOS)
    x_n = ds.GraphSignal.add_noise(x, n_p)
    for i in range(N_SCENARIOS):
        dec = GraphDeepDecoder(n_chans, sizes[i], Us[i], As=As[i],
                               ups=EXPS[i][0], gamma=EXPS[i][1],
                               batch_norm=batch_norm, act_fn=act_fun,
                               last_act_fn=last_act_fun)

        # Implement model??
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
        print('{}. {} '.format(i+1, EXPS[i]))
        print('\tMean MSE: {}\tMedian MSE: {}\tSTD: {}'
              .format(mean_mse[i], median_mse[i], std[i]))


def save_partial_results(error, G_params, n_p):
    if not os.path.isdir('./results/test_ups'):
        os.makedirs('./results/test_ups')

    data = {'SEED': SEED, 'EXPS': EXPS, 't': t, 'n_p': n_p,
            'n_signals': n_signals, 'L': L, 'batch_norm': batch_norm,
            'c_method': c_method, 'alg': alg, 'last_act_fun': last_act_fun,
            'G_params': G_params, 'n_chans': n_chans,
            'error': error}
    timestamp = datetime.datetime.now().strftime("%Y_%m_%d-%H_%M")
    path = './results/test_ups/ups_pn_{}_{}'.format(n_p, timestamp)
    np.save(path, data)
    print('SAVED as:', path)


def save_results(error, G_params):
    if not os.path.isdir('./results/test_ups'):
        os.makedirs('./results/test_ups')

    data = {'SEED': SEED, 'EXPS': EXPS, 't': t, 'N_P': N_P,
            'n_signals': n_signals, 'L': L, 'batch_norm': batch_norm,
            'c_method': c_method, 'alg': alg, 'last_act_fun': last_act_fun,
            'G_params': G_params, 'n_chans': n_chans,
            'error': error}
    timestamp = datetime.datetime.now().strftime("%Y_%m_%d-%H_%M")
    path = './results/test_ups/ups_{}'.format(timestamp)
    np.save(path, data)
    print('SAVED as:', path)


if __name__ == '__main__':
    # Graph parameters
    G_params = {}
    G_params['type'] = ds.SBM  # SBM or ER
    G_params['N'] = N = 256
    G_params['k'] = 4
    G_params['p'] = 0.15
    G_params['q'] = 0.01/4
    G_params['type_z'] = ds.RAND

    # Set seeds
    np.random.seed(SEED)
    manual_seed(SEED)

    G = ds.create_graph(G_params, SEED)
    sizes, Us, As = compute_clusters(G_params['k'])

    print("CPUs used:", N_CPUS)
    start_time = time.time()
    error = np.zeros((len(N_P), n_signals, N_SCENARIOS))
    for i, n_p in enumerate(N_P):
        print('Noise:', n_p)
        results = []
        with Pool(processes=N_CPUS) as pool:
            for j in range(n_signals):
                signal = ds.DifussedSparseGS(G, L, G_params['k'])
                signal.signal_to_0_1_interval()
                signal.to_unit_norm()

                results.append(pool.apply_async(test_upsampling,
                               args=[j, signal.x, sizes, Us, As, n_p]))

            for j in range(n_signals):
                error[i, j, :] = results[j].get()

        # Print result:
        print_results(error[i, :, :])
    print('--- {} hours ---'.format((time.time()-start_time)/3600))
    if SAVE:
        save_results(error, G_params)

import sys
import os
import time
import datetime
from multiprocessing import Pool, cpu_count

from scipy.sparse.csgraph import dijkstra
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
from torch import manual_seed

sys.path.insert(0, '../graph_deep_decoder')
from graph_deep_decoder import datasets as ds
from graph_deep_decoder import utils
from graph_deep_decoder import graph_clustering as gc
from graph_deep_decoder.architecture import GraphDeepDecoder


# Constants
N_SIGNALS = 2  # 100
SEED = 15
N_P = [0]  # [0, .1, .2, .3, .4, .5]
SAVE = True
N_CPUS = cpu_count()

INPUTS = [ds.LINEAR, ds.MEDIAN]
EXPERIMENTS = ['bandlimited',
               {'ups': gc.REG, 'arch': [3, 3, 3, 3, 3, 1],
                't': [256]*6},
               {'ups': gc.WEI, 'arch': [3, 3, 3, 3, 3, 1],
                't': [4, 16, 64, 256, 256, 256]}]
N_EXPS = len(INPUTS)*len(EXPERIMENTS)


def compute_clusters(G, root_clust):
    sizes = []
    Us = []
    As = []
    for exp in EXPERIMENTS:
        if not isinstance(exp, dict):
            sizes.append(None)
            Us.append(None)
            As.append(None)
            continue

        exp['t'][-1] = G.N
        cluster = gc.MultiResGraphClustering(G, exp['t'], root_clust)
        sizes.append(cluster.sizes)
        Us.append(cluster.Us)
        As.append(cluster.As)
    return sizes, Us, As


def create_signal(signal_type, G, L, k, D):
    signal = ds.GraphSignal.create_graph_signal(signal_type, G, L, k, D)
    signal.signal_to_0_1_interval()
    signal.to_unit_norm()
    return signal


def test_input(id, signals, sizes, Us, As, n_p,
               last_act_fn, batch_norm, V):
    error = np.zeros(N_EXPS)

    for i, x in enumerate(signals):
        x_n = ds.GraphSignal.add_noise(x, n_p)
        for j, exp in enumerate(EXPERIMENTS):
            cont = i*len(EXPERIMENTS)+j
            if not isinstance(exp, dict):
                x_est = utils.bandlimited_model(x_n, V, n_coefs=63)
            else:
                dec = GraphDeepDecoder(exp['arch'], sizes[j], Us[j], As=As[j],
                                       ups=exp['ups'], batch_norm=batch_norm,
                                       last_act_fn=last_act_fn)
                x_est, _ = dec.fit(x_n, n_iter=3000)
            error[cont] = np.sum(np.square(x-x_est))/np.square(np.linalg.norm(x))
            print('Signal: {} Scenario: {} Error: {:.4f}'
                  .format(id, cont+1, error[cont]))
    return error


def print_results(error, n_p):
    mean_mse = np.mean(error, axis=0)
    median_mse = np.median(error, axis=0)
    std = np.std(error, axis=0)
    for i, s_in in enumerate(INPUTS):
        for j, exp in enumerate(EXPERIMENTS):
            cont = i*len(EXPERIMENTS)+j
            print('{}. INPUT: {} EXP: {}'.format(cont+1, s_in, exp))
            print('\tMean MSE: {}\tMedian MSE: {}\tSTD: {}'
                            .format(mean_mse[cont], median_mse[cont], std[cont]))


def save_partial_results(data, n_p):
    data['n_p'] = n_p
    if not os.path.isdir('./results/test_input'):
        os.makedirs('./results/test_input')
    timestamp = datetime.datetime.now().strftime("%Y_%m_%d-%H_%M")
    path = './results/test_input/input_n_p_{}_{}'.format(n_p, timestamp)
    np.save(path, data)
    print('SAVED as:', path)


def save_results(data):
    data['N_P'] = N_P
    if not os.path.isdir('./results/test_input'):
        os.makedirs('./results/test_input')
    timestamp = datetime.datetime.now().strftime("%Y_%m_%d-%H_%M")
    path = './results/test_input/input_{}'.format(timestamp)
    np.save(path, data)
    print('SAVED as:', path)


if __name__ == '__main__':
    data = {}
    data['L'] = L = 6
    data['last_act_fn'] = nn.Sigmoid()
    data['batch_norm'] = True
    data['EXPERIMENTS'] = EXPERIMENTS
    data['INPUTS'] = INPUTS
    
    # Graph parameters
    G_params = {}
    G_params['type'] = ds.SBM  # SBM or ER
    G_params['N'] = N = 256
    G_params['k'] = k = 4
    G_params['p'] = 0.15
    G_params['q'] = 0.01/k
    G_params['type_z'] = ds.RAND

    # Set seeds
    np.random.seed(SEED)
    manual_seed(SEED)

    start_time = time.time()
    G = ds.create_graph(G_params, seed=SEED)
    G.compute_fourier_basis()
    D = dijkstra(G.W)
    sizes, Us, As = compute_clusters(G, G_params['k'])
    data['g_params'] = G_params

    print("CPUs used:", N_CPUS)
    error = np.zeros((len(N_P), N_SIGNALS, N_EXPS))
    for i, n_p in enumerate(N_P):
        print('Noise:', n_p)
        results = []
        with Pool(processes=N_CPUS) as pool:
            for j in range(N_SIGNALS):
                signals = [create_signal(s_in, G, L, k, D).x for s_in in INPUTS]
                results.append(pool.apply_async(test_input,
                               args=[j, signals, sizes, Us, As, n_p,
                                     data['last_act_fn'], data['batch_norm'],
                                     G.U]))
            for j in range(N_SIGNALS):
                error[i, j, :] = results[j].get()

        data['error'] = error[i, :, :]
        print_results(error[i, :, :], n_p)

    data['error'] = error
    if SAVE:
        save_results(data)
    print('--- {} hours ---'.format((time.time()-start_time)/3600))

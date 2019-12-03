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
from graph_deep_decoder.model import Model
from graph_deep_decoder import utils


# Constants
N_CPUS = cpu_count()
SAVE = True
PATH = './results/arch'
FILE_PREF = 'arch_'

SEED = 15
N_P = [0, .2, .4, .6]
EXPS = [{'fts': [10]*7 + [1], 'nodes': [4, 8, 16, 32, 64] + [256]*3,
         'epochs': 5000},
        {'fts': [10]*7 + [1], 'nodes': [4, 8, 16, 32, 64] + [256]*3,
         'epochs': 300},
        {'fts': [7]*6 + [1], 'nodes': [4, 8, 16, 32] + [256]*3,
         'epochs': 5000},
        {'fts': [7]*6 + [1], 'nodes': [4, 8, 16, 32] + [256]*3,
         'epochs': 300},
        {'fts': [3]*5 + [1], 'nodes': [4, 16, 32] + [256]*3, 'epochs': 5000},
        {'fts': [3]*5 + [1], 'nodes': [4, 16, 32] + [256]*3, 'epochs': 300},
        # {'fts': [2]*8 + [1], 'nodes': [4, 8, 16, 32, 64, 128] + [256]*3,
        #  'epochs': 5000},
        # {'fts': [2]*8 + [1], 'nodes': [4, 8, 16, 32, 64, 128] + [256]*3,
        #  'epochs': 500}
        ]

N_EXPS = len(EXPS)


def plot_clusters(G, cluster):
    G.set_coordinates(kind='community2D')
    _, axes = plt.subplots(1, 2)
    G.plot_signal(cluster.labels[0], ax=axes[0])
    axes[1].spy(G.W)
    plt.show()


def compute_clusters(G, k):
    sizes = []
    Us = []
    hier_As = []
    for i in range(N_EXPS):
        cluster = gc.MultiResGraphClustering(G, EXPS[i]['nodes'], k)
        sizes.append(cluster.sizes)
        Us.append(cluster.Us)
        hier_As.append(cluster.As)
    return sizes, Us, hier_As


def run(id, x, sizes, Us, As, n_p):
    err = np.zeros(N_EXPS)
    node_err = np.zeros(N_EXPS)
    params = np.zeros(N_EXPS)
    x_n = ds.GraphSignal.add_noise(x, n_p)
    for i in range(N_EXPS):
        dec = GraphDeepDecoder(EXPS[i]['fts'], sizes[i], Us[i],
                               As=As[i])
        model = Model(dec, learning_rate=0.001, decay_rate=1,
                      epochs=EXPS[i]['epochs'])
        model.fit(x_n, x=x)
        node_err[i], err[i] = model.test(x)
        params[i] = model.count_params()
        print('Signal {}-{} ({}):\tEpochs: {}\tNode Err: {:.8f}\tErr: {:.6f}'
              .format(id, i+1, params[i], EXPS[i]['epochs'], node_err[i],
                      err[i]))
    return node_err, err, params


def save_results(error, n_params, G_params):
    if not os.path.isdir('./results/test_arch'):
        os.makedirs('./results/test_arch')

    data = {'SEED': SEED, 'EXPERIMENTS': EXPS,
            'N_P': N_P, 'G_params': G_params,
            'error': error, 'n_params': n_params}
    timestamp = datetime.datetime.now().strftime("%Y_%m_%d-%H_%M")
    path = './results/test_arch/arch_{}'.format(timestamp)
    np.save(path, data)
    print('SAVED as:', path)


def create_legend(params):
    legend = []
    for i, exp in enumerate(EXPS):
        legend.append('P: {}, E: {}'
                      .format(params[i], exp['epochs']))
    return legend


if __name__ == '__main__':
    # Set random seed
    np.random.seed(SEED)
    manual_seed(SEED)

    # Graph parameters
    Gs = {}
    Gs['type'] = ds.SBM  # SBM or ER
    Gs['N'] = 256
    Gs['k'] = 4
    Gs['p'] = 0.25
    Gs['q'] = [[0, 0.0075, 0, 0.0],
               [0.0075, 0, 0.004, 0.0025],
               [0, 0.004, 0, 0.005],
               [0, 0.0025, 0.005, 0]]
    Gs['type_z'] = ds.RAND

    # Signal parameters
    Signals = {}
    Signals['n_signals'] = 50
    Signals['deltas'] = Gs['k']
    Signals['L'] = 6
    Signals['Noise'] = N_P

    # Tuning parameters --> NOT USED
    batch_norm = True
    act_fun = nn.ReLU()
    up_method = gc.WEI
    gamma = 0.5
    last_act_fun = nn.Sigmoid()

    # NOTE: chage for using more graphs!
    G = ds.create_graph(Gs, SEED)
    sizes, Us, As = compute_clusters(G, Gs['k'])

    print("CPUs used:", N_CPUS)
    start_time = time.time()
    err = np.zeros((len(N_P), Signals['n_signals'], N_EXPS))
    node_err = np.zeros((len(N_P), Signals['n_signals'], N_EXPS))
    for i, n_p in enumerate(N_P):
        print('Noise:', n_p)
        results = []
        with Pool(processes=N_CPUS) as pool:
            for j in range(Signals['n_signals']):
                data = ds.DifussedSparseGS(G, Signals['L'],
                                           Signals['deltas'])
                data.to_unit_norm()
                results.append(pool.apply_async(run,
                               args=[j, data.x, sizes, Us, As, n_p]))
            for j in range(Signals['n_signals']):
                node_err[i, j, :], err[i, j, :], n_params = results[j].get()

        # Print result:
        utils.print_partial_results(node_err[i, :, :], err[i, :, :], n_params)

    print('--- {} hours ---'.format((time.time()-start_time)/3600))
    utils.print_results(node_err, err, N_P, n_params)
    utils.plot_results(err, N_P, legend=create_legend(n_params))
    if SAVE:
        data = {
            'seed': SEED,
            'exps': EXPS,
            'Gs': Gs,
            'Signals': Signals,
            # Faltaría algo para learning, model, etc...
            'node_err': node_err,
            'err': err,
        }
        utils.save_results(FILE_PREF, PATH, data)

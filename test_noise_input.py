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
import torch.nn as nn
from torch import manual_seed

sys.path.insert(0, '../graph_deep_decoder')
from graph_deep_decoder import datasets as ds
from graph_deep_decoder.graph_clustering import Type_A, MultiResGraphClustering
from graph_deep_decoder.architecture import GraphDeepDecoder, Ups
from graph_deep_decoder.model import Model
from graph_deep_decoder import utils


# Constants
N_CPUS = cpu_count()
SAVE = True
PATH = './results/noise/'
FILE_PREF = 'noise_'

SEED = 15
MAX_EPOCHS = 10000  # 5000
EXPS = [
        # {'ups': Ups.NONE, 'gamma': None, 'fts': [15]*5 + [1],
        #  'type_A': Type_A.NONE,  'K': 0, 'fmt': '-'},
        {'ups': Ups.U_MAT, 'gamma': 0.5, 'type_A': Type_A.NONE,
         'fts': [15]*5 + [1],  'K': 0, 'fmt': '-'},
        {'ups': Ups.U_MEAN, 'gamma': 0.5, 'type_A': Type_A.WEI,
         'fts': [15]*5 + [1],  'K': 0, 'fmt': '-'},
        {'ups': Ups.GF, 'gamma': None,
         'type_A': Type_A.WEI, 'fts': [15]*5 + [1], 'K': 3, 'fmt': '-'},

        # {'ups': Ups.NONE, 'gamma': None, 'fts': [6]*5 + [1],
        #  'type_A': Type_A.NONE,  'K': 0, 'fmt': '--'},
        {'ups': Ups.U_MAT, 'gamma': 0.5, 'type_A': Type_A.NONE,
         'fts': [6]*5 + [1],  'K': 0, 'fmt': '--'},
        {'ups': Ups.U_MEAN, 'gamma': 0.5, 'type_A': Type_A.WEI,
         'fts': [6]*5 + [1],  'K': 0, 'fmt': '--'},
        {'ups': Ups.GF, 'gamma': None,
         'type_A': Type_A.WEI, 'fts': [6]*5 + [1], 'K': 3, 'fmt': '--'},
       ]

N_EXPS = len(EXPS)


def compute_clusters(G, sizes, root_clusts):
    clusters = []
    for exp in EXPS:
        cluster = MultiResGraphClustering(G, sizes, root_clusts,
                                          type_A=exp['type_A'])
        clusters.append(cluster)
    return clusters


def run(id, Gs, Net):
    G = ds.create_graph(Gs, SEED)
    clts = compute_clusters(G, Net['nodes'], Gs['k'])
    signal = ds.DeterministicGS(G, np.random.randn(G.N))
    signal.to_unit_norm()

    err = np.zeros((N_EXPS, MAX_EPOCHS))
    params = np.zeros(N_EXPS)
    for i, exp in enumerate(EXPS):
        if exp['ups'] is Ups.NONE or exp['ups'] is None:
            clts[i].sizes = [G.N]*len(exp['fts'])

        dec = GraphDeepDecoder(exp['fts'], clts[i].sizes, clts[i].Us,
                               As=clts[i].As, act_fn=Net['af'], ups=exp['ups'],
                               batch_norm=Net['bn'], last_act_fn=Net['laf'])

        model = Model(dec, learning_rate=Net['lr'], epochs=MAX_EPOCHS)
        err[i, :], _, _ = model.fit(signal.x)
        node_test_err, test_err = model.test(signal.x)
        params[i] = model.count_params()
        print('Graph {}-{} ({}):\tErr: {:.8f}\tTest Err: {:.8f}'
              .format(id, i+1, params[i], err[i, -1], node_test_err))
    # Multiplying by N because we're interested in the error of the whole
    # signal, # not only the node error
    return err.T*G.N, params


def create_legend(params):
    legend = []
    for i, exp in enumerate(EXPS):
        legend.append('Ups: {}, A: {}, G: {}, K: {}, P: {}'
                      .format(exp['ups'].name, exp['type_A'].name,
                              exp['gamma'], exp['K'], params[i]))
    return legend


if __name__ == '__main__':
    # Set random seed
    np.random.seed(SEED)
    manual_seed(SEED)

    # Graph parameters
    Gs = {}
    Gs['type'] = ds.SBM  # SBM or ER
    Gs['n_graphs'] = 15
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
    Signals['n_signals'] = 1

    # Net
    Net = {}
    Net['laf'] = nn.Tanh()
    Net['af'] = nn.Tanh()
    Net['bn'] = True
    Net['lr'] = 0.005
    Net['nodes'] = [4, 16, 32] + [256]*3
    Net['epochs'] = MAX_EPOCHS

    # print("CPUs used:", N_CPUS)
    start_time = time.time()
    err = np.zeros((MAX_EPOCHS, Gs['n_graphs'], N_EXPS))
    for i in range(Gs['n_graphs']):
        err[:, i, :], params = run(i, Gs, Net)

    # Multi CPU version
    # with Pool(processes=N_CPUS) as pool:
    #     for i in range(Gs['n_graphs']):
    #         results = []
    #         results.append(pool.apply_async(run, args=[i, Gs, Net]))
    #     for i in range(Gs['n_graphs']):
    #         # err[:, i, :], params = results[i].get()
    #         abc, params = results[i].get()

    print('--- {} hours ---'.format((time.time()-start_time)/3600))
    legend = create_legend(params)
    fmts = [exp['fmt'] for exp in EXPS]
    epochs = np.arange(MAX_EPOCHS)
    # Plot median of all realiztions
    utils.plot_results(err, epochs, legend=legend, fmts=fmts)
    # Plot only first realization
    first_err = err[:, 0, :].reshape([MAX_EPOCHS, 1, N_EXPS])
    utils.plot_results(first_err, epochs, legend=legend, fmts=fmts)
    if SAVE:
        data = {
            'seed': SEED,
            'exps': EXPS,
            'Gs': Gs,
            'Signals': Signals,
            'Net': Net,
            'node_err': None,
            'err': err,
            'params': params,
            'legend': legend,
            'fmts': fmts,
        }
        utils.save_results(FILE_PREF, PATH, data)

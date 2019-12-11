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
import torch.nn as nn
from torch import manual_seed

sys.path.insert(0, '../graph_deep_decoder')
from graph_deep_decoder import datasets as ds
from graph_deep_decoder.graph_clustering import Ups, MultiResGraphClustering
from graph_deep_decoder.architecture import GraphDeepDecoder
from graph_deep_decoder.model import Model
from graph_deep_decoder import utils

# Constants
N_CPUS = cpu_count()
SAVE = True
PATH = './results/ups/'
FILE_PREF = 'ups_'

SEED = 15
N_P = [0, .1, .2, .3, .4, .5]
EXPS = [{'ups': Ups.NONE, 'gamma': None, 'nodes': [256]*6, 'fts': [15]*5 + [1],
         'fmt': 'v-'},
        {'ups': Ups.REG, 'gamma': None, 'nodes': [4, 16, 32] + [256]*3,
         'fts': [15]*5 + [1], 'fmt': '^-'},
        {'ups': Ups.NO_A, 'gamma': None, 'nodes': [4, 16, 32] + [256]*3,
         'fts': [15]*5 + [1], 'fmt': 'P-'},
        {'ups': Ups.BIN, 'gamma': 0.5, 'nodes': [4, 16, 32] + [256]*3,
         'fts': [15]*5 + [1], 'fmt': 'X-'},
        {'ups': Ups.WEI, 'gamma': 0.5, 'nodes': [4, 16, 32] + [256]*3,
         'fts': [15]*5 + [1], 'fmt': 'o-'},

        {'ups': Ups.WEI, 'gamma': 0, 'nodes': [4, 16, 32] + [256]*3,
         'fts': [15]*5 + [1], 'fmt': 'o-.'},
        {'ups': Ups.WEI, 'gamma': 0.25, 'nodes': [4, 16, 32] + [256]*3,
         'fts': [15]*5 + [1], 'fmt': 'o:'},
        {'ups': Ups.WEI, 'gamma': 0.75, 'nodes': [4, 16, 32] + [256]*3,
         'fts': [15]*5 + [1], 'fmt': 'o--'},

        # {'ups': Ups.NONE, 'gamma': None, 'nodes': [256]*4, 'fts': [15]*3 + [1],
        #  'fmt': 'v--'},
        # {'ups': Ups.REG, 'gamma': None, 'nodes': [4] + [256]*3,
        #  'fts': [15]*3 + [1], 'fmt': '^--'},
        # {'ups': Ups.NO_A, 'gamma': None, 'nodes': [4] + [256]*3,
        #  'fts': [15]*3 + [1], 'fmt': 'P--'},
        # {'ups': Ups.BIN, 'gamma': 0.5, 'nodes': [4] + [256]*3,
        #  'fts': [15]*3 + [1], 'fmt': 'X--'},
        # {'ups': Ups.WEI, 'gamma': 0.5, 'nodes': [4] + [256]*3,
        #  'fts': [15]*3 + [1], 'fmt': 'o--'}
       ]
N_EXPS = len(EXPS)


def compute_clusters(G, root_clusts):
    clusters = []
    for exp in EXPS:
        cluster = MultiResGraphClustering(G, exp['nodes'], root_clusts,
                                          up_method=exp['ups'])
        clusters.append(cluster)
    return clusters


def run(id, Gs, Signals, Net, n_p):
    G = ds.create_graph(Gs, SEED)
    clts = compute_clusters(G, Gs['k'])
    signal = ds.GraphSignal.create(Signals['type'], G, Signals['non_lin'],
                                   Signals['L'], Signals['deltas'],
                                   to_0_1=Signals['to_0_1'])
    x_n = ds.GraphSignal.add_noise(signal.x, n_p)

    err = np.zeros(N_EXPS)
    node_err = np.zeros(N_EXPS)
    params = np.zeros(N_EXPS)
    for i, exp in enumerate(EXPS):
        dec = GraphDeepDecoder(exp['fts'], clts[i].sizes, clts[i].Us,
                               As=clts[i].As, act_fn=Net['af'], ups=exp['ups'],
                               gamma=exp['gamma'], batch_norm=Net['bn'],
                               last_act_fn=Net['laf'])

        model = Model(dec, learning_rate=Net['lr'], decay_rate=Net['dr'],
                      epochs=Net['epochs'])
        model.fit(x_n)
        node_err[i], err[i] = model.test(signal.x)
        params[i] = model.count_params()
        print('Graph {}-{} ({}):\tNode Err: {:.8f}\tErr: {:.6f}'
              .format(id, i+1, params[i], node_err[i], err[i]))
    return node_err, err, params


def create_legend(params):
    legend = []
    for i, exp in enumerate(EXPS):
        legend.append('Ups: {}, G: {}, P: {}'
                      .format(exp['ups'].name, exp['gamma'], params[i]))
    return legend


if __name__ == '__main__':
    # Set random seed
    np.random.seed(SEED)
    manual_seed(SEED)

    # Graph parameters
    Gs = {}
    Gs['type'] = ds.SBM  # SBM or ER
    Gs['n_graphs'] = 50
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
    Signals['type'] = ds.SigType.DS
    Signals['non_lin'] = ds.NonLin.MEDIAN
    Signals['deltas'] = Gs['k']
    Signals['L'] = 6
    Signals['noise'] = N_P
    Signals['to_0_1'] = False

    Net = {}
    Net['laf'] = nn.Tanh()
    Net['af'] = nn.Tanh()  # nn.CELU()
    Net['bn'] = True
    Net['lr'] = 0.005
    Net['dr'] = 1
    Net['epochs'] = 250

    print("CPUs used:", N_CPUS)
    start_time = time.time()
    err = np.zeros((len(N_P), Gs['n_graphs'], N_EXPS))
    node_err = np.zeros((len(N_P), Gs['n_graphs'], N_EXPS))
    for i, n_p in enumerate(N_P):
        print('Noise:', n_p)
        results = []
        with Pool(processes=N_CPUS) as pool:
            for j in range(Gs['n_graphs']):
                results.append(pool.apply_async(run,
                               args=[j, Gs, Signals, Net, n_p]))
            for j in range(Gs['n_graphs']):
                node_err[i, j, :], err[i, j, :], params = results[j].get()

        utils.print_partial_results(node_err[i, :, :], err[i, :, :], params)

    utils.print_results(node_err, err, N_P, params)
    print('--- {} hours ---'.format((time.time()-start_time)/3600))
    legend = create_legend(params)
    fmts = [exp['fmt'] for exp in EXPS]
    utils.plot_results(err, N_P, legend=legend, fmts=fmts)
    if SAVE:
        data = {
            'seed': SEED,
            'exps': EXPS,
            'Gs': Gs,
            'Signals': Signals,
            'Net': Net,
            'node_err': node_err,
            'err': err,
            'params': params,
            'legend': legend,
            'fmts': fmts,
        }
        utils.save_results(FILE_PREF, PATH, data)

import sys
import os
import time
import datetime
from multiprocessing import Pool, cpu_count

import numpy as np
import torch.nn as nn
from torch import manual_seed

# sys.path.insert(0, '../graph_deep_decoder')
from graph_deep_decoder import datasets as ds
from graph_deep_decoder.graph_clustering import Type_A, MultiResGraphClustering
from graph_deep_decoder.architecture import GraphDeepDecoder, Ups
from graph_deep_decoder.model import Model
from graph_deep_decoder import utils

# Constants
N_CPUS = cpu_count()
SAVE = True
PATH = './results/graph/'
FILE_PREF = 'graph_'

SEED = 15
N_P = [0, .1, .2, .3, .4, .5]

EXPS = [{'ups': Ups.U_MEAN, 'gamma': 0.5, 'nodes': [4, 16, 64] + [256]*3,
         'type_A': Type_A.WEI, 'fts': [15]*5 + [1],  'K': None, 'fmt': 'o'},
        {'ups': Ups.U_MEAN, 'gamma': 0.5, 'nodes': [4, 16, 64] + [256]*3,
         'type_A': Type_A.WEI, 'fts': [3]*5 + [1],  'K': None, 'fmt': 'X'},
        {'ups': Ups.GF, 'gamma': None, 'nodes': [4, 16, 64] + [256]*3,
         'type_A': Type_A.WEI, 'fts': [15]*5 + [1], 'K': 3, 'fmt': 'P'},
        ]
GRAPHS = [
            {'type': ds.SBM, 'N': 256, 'k': 4, 'p': 0.25, 'type_z': ds.RAND,
             'q': [[0, 0.0075, 0, 0.0],
                   [0.0075, 0, 0.004, 0.0025],
                   [0, 0.004, 0, 0.005],
                   [0, 0.0025, 0.005, 0]],
             'fmt': '-'},
            {'type': ds.ER, 'N': 256, 'p': 0.05, 'k': 4, 'fmt': '--'},
            {'type': ds.BA, 'N': 256, 'm0': 2, 'm': 2, 'k': 4, 'fmt': ':'}
         ]
N_EXPS = len(GRAPHS)*len(EXPS)


def compute_clusters(G, root_clusts):
    clusters = []
    for exp in EXPS:
        cluster = MultiResGraphClustering(G, exp['nodes'], root_clusts,
                                          type_A=exp['type_A'])
        clusters.append(cluster)
    return clusters


def run(id, Signals, Net, n_p):
    node_err = np.zeros(N_EXPS)
    err = np.zeros(N_EXPS)
    params = np.zeros(N_EXPS)
    for i, Gs in enumerate(GRAPHS):
        G = ds.create_graph(Gs, SEED)
        clts = compute_clusters(G, Gs['k'])
        signal = ds.GraphSignal.create(Signals['type'], G, Signals['non_lin'],
                                       Signals['L'])
        x_n = ds.GraphSignal.add_noise(signal.x, n_p)
        for j, exp in enumerate(EXPS):
            cont = i*len(EXPS)+j
            dec = GraphDeepDecoder(exp['fts'], clts[j].sizes, clts[j].Us,
                                   As=clts[j].As, ups=exp['ups'],
                                   act_fn=Net['af'], K=exp['K'],
                                   last_act_fn=Net['laf'], gamma=exp['gamma'],
                                   batch_norm=Net['bn'])
            model = Model(dec, learning_rate=Net['lr'], epochs=Net['epochs'])

            params[cont] = model.count_params()
            model.fit(x_n)
            node_err[cont], err[cont] = model.test(signal.x)
            print('Signal {}-{} ({}):  Eps: {}  Node Err: {:.8f}  Err: {:.6f}'
                  .format(id, cont+1, params[cont], Net['epochs'],
                          node_err[cont], err[cont]))
    return node_err, err, params


def create_legend(params):
    legend = []
    for graph in GRAPHS:
        sig_txt = 'Unkown, '
        if graph['type'] == ds.SBM:
            sig_txt = 'SBM, '
        elif graph['type'] == ds.ER:
            sig_txt = 'ER, '
        elif graph['type'] == ds.BA:
            sig_txt = 'BA, '
        for j, exp in enumerate(EXPS):
            txt = sig_txt + '{}, P: {}'.format(exp['ups'], params[j])
            legend.append(txt)
    return legend


def create_fmts():
    fmts = []
    for graph in GRAPHS:
        for exp in EXPS:
            fmts.append(exp['fmt'] + graph['fmt'])
    return fmts


if __name__ == '__main__':
    # Set random seed
    np.random.seed(SEED)
    manual_seed(SEED)

    # Signal parameters
    Signals = {}
    Signals['n_signals'] = 50
    Signals['type'] = ds.SigType.DW
    Signals['non_lin'] = ds.NonLin.SQ2
    Signals['L'] = L = 6
    Signals['noise'] = N_P

    # Network parameters
    Net = {}
    Net['laf'] = nn.Tanh()
    Net['af'] = nn.Tanh()
    Net['bn'] = True
    Net['lr'] = 0.005
    Net['epochs'] = 3500

    print("CPUs used:", N_CPUS)
    start_time = time.time()
    err = np.zeros((len(N_P), Signals['n_signals'], N_EXPS))
    node_err = np.zeros((len(N_P), Signals['n_signals'], N_EXPS))
    for i, n_p in enumerate(N_P):
        print('Noise:', n_p)
        results = []
        with Pool(processes=N_CPUS) as pool:
            for j in range(Signals['n_signals']):
                results.append(pool.apply_async(run,
                               args=[j, Signals, Net, n_p]))
            for j in range(Signals['n_signals']):
                node_err[i, j, :], err[i, j, :], params = results[j].get()

        utils.print_partial_results(node_err[i, :, :], err[i, :, :], params)

    utils.print_results(node_err, err, N_P, params)
    print('--- {} hours ---'.format((time.time()-start_time)/3600))
    legend = create_legend(params)
    fmts = create_fmts()
    utils.plot_results(err, N_P, legend=legend, fmts=fmts)
    if SAVE:
        data = {
            'seed': SEED,
            'exps': EXPS,
            'Gs': None,
            'Signals': Signals,
            'Net': Net,
            'node_err': node_err,
            'err': err,
            'params': params,
            'legend': legend,
            'fmts': fmts,
        }
        utils.save_results(FILE_PREF, PATH, data)
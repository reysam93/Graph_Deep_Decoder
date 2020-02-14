import datetime
import os
import sys
import time
from multiprocessing import Pool, cpu_count

import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
from pygsp.graphs import Graph
from scipy.io import loadmat
from torch import manual_seed

sys.path.insert(0, 'graph_deep_decoder')
from graph_deep_decoder import datasets as ds
from graph_deep_decoder.graph_clustering import Type_A, MultiResGraphClustering
from graph_deep_decoder.architecture import GraphDeepDecoder, Ups
from graph_deep_decoder.model import Model, BLModel
from graph_deep_decoder import utils

# Constants
N_CPUS = cpu_count()
SAVE = True
PATH = './results/denoise_real/'
FILE_PREF = 'denoise_real_'

SEED = 15
DATASET_PATH = 'dataset/graphs.mat'
MAX_SIGNALS = 25  # 100
MIN_SIZE = 50
MAX_SIZE = 150
MAX_SM = 3
ATTR = 6
N_P = [0, .1, .2, .3]  # [0, .1, .2, .3, .4, .5]

# Try removing None Layers
# Con 30 params ha ganado [10, 25] + [None]*3, REPETIR
EXPS = [
        # Mejores resultado
        # {'type': 'DD', 'A': Type_A.WEI, 'ups': Ups.U_MEAN, 'fts': [15]*4 + [1], 'K': 0,
        #  'nodes': [32, 46] + [None]*3, 'gamma': 0.5, 'epochs': 1000, 'fmt': 'v-'},
        # {'type': 'DD', 'A': Type_A.WEI, 'ups': Ups.U_MEAN, 'fts': [15]*4 + [1], 'gamma': 0.5,
        #  'nodes': [10, 25] + [None]*3, 'K': 0, 'epochs': 1000, 'fmt': '^-'},
        # {'type': 'DD', 'A': Type_A.NONE, 'ups': Ups.NONE, 'fts': [15]*4 + [1], 'gamma': 0.5,
        #  'nodes': [None]*5, 'epochs': 1000, 'fmt': '<-'},
        # {'type': 'DD', 'A': Type_A.WEI, 'ups': Ups.U_MEAN, 'fts': [3]*4 + [1], 'gamma': 0.5,
        #  'nodes': [24, 48] + [None]*3, 'K': 0,  'epochs': 1000, 'fmt': 'o-'},


        {'type': 'DD', 'A': Type_A.NONE, 'ups': Ups.NONE, 'fts': [6]*4 + [1], 'gamma': 0.5,
         'nodes': [None]*5, 'K': 0, 'epochs': 1500, 'fmt': '<-'},

        {'type': 'DD', 'A': Type_A.WEI, 'ups': Ups.GF, 'fts': [6]*5 + [1], 'K': 3,
         'nodes': [4, 16, 32] + [None]*3, 'gamma': 0.5, 'epochs': 1500, 'fmt': 'o-'},
        {'type': 'DD', 'A': Type_A.WEI, 'ups': Ups.U_MEAN, 'fts': [6]*5 + [1], 'K': 2,
         'nodes': [4, 16, 32] + [None]*3, 'gamma': 0.5, 'epochs': 1500, 'fmt': 'o--'},
        
        {'type': 'DD', 'A': Type_A.WEI, 'ups': Ups.GF, 'fts': [6]*3 + [1], 'K': 5,
         'nodes': [10] + [None]*3, 'gamma': 0.5, 'epochs': 1500, 'fmt': 'o-'},
        {'type': 'DD', 'A': Type_A.WEI, 'ups': Ups.U_MEAN, 'fts': [6]*3 + [1], 'K': 2,
         'nodes': [10] + [None]*3, 'gamma': 0.5, 'epochs': 1500, 'fmt': 'o--'},

        {'type': 'DD', 'A': Type_A.WEI, 'ups': Ups.GF, 'fts': [6]*4 + [1], 'K': 3,
         'nodes': [16, 32] + [None]*3, 'gamma': 0.5, 'epochs': 1500, 'fmt': 'o-'},
        {'type': 'DD', 'A': Type_A.WEI, 'ups': Ups.U_MEAN, 'fts': [6]*4 + [1], 'K': 2,
         'nodes': [16, 32] + [None]*3, 'gamma': 0.5, 'epochs': 1500, 'fmt': 'o--'},
        ]
N_EXPS = len(EXPS)


def compute_clusters(Gs):
    sizes = []
    Us = []
    As = []
    for i, G in enumerate(Gs):
        sizes.append([])
        Us.append([])
        As.append([])
        for exp in EXPS:
            if exp['type'] == 'BL':
                sizes[i].append(None)
                Us[i].append(None)
                As[i].append(None)
                continue

            nodes = [G.N if n is None else n for n in exp['nodes']]
            cluster = MultiResGraphClustering(G, nodes, nodes[0],
                                              type_A=exp['A'])
            sizes[i].append(cluster.sizes)
            Us[i].append(cluster.Us)
            As[i].append(cluster.As)
    return sizes, Us, As


def run(id, x, sizes, Us, As, n_p, Net, V):
    node_err = np.zeros(N_EXPS)
    err = np.zeros(N_EXPS)
    params = np.zeros(N_EXPS)

    x_n = ds.GraphSignal.add_noise(x, n_p)

    for i, exp in enumerate(EXPS):
        # Construct model
        if exp['type'] == 'BL':
            model = BLModel(V, exp['params'])
        else:
            dec = GraphDeepDecoder(exp['fts'], sizes[i], Us[i],
                                   As=As[i], ups=exp['ups'], K=exp['K'],
                                   gamma=exp['gamma'], act_fn=Net['af'],
                                   last_act_fn=Net['laf'],
                                   batch_norm=Net['bn'])
            model = Model(dec, learning_rate=Net['lr'],
                          epochs=exp['epochs'])
        params[i] = model.count_params()
        model.fit(x_n)
        node_err[i], err[i] = model.test(x)
        print('Graph {}-{} ({}): N: {} Err: {:.6f}'
              .format(id, i+1, params[i], x_n.shape[0],
                      err[i]))
    return node_err, err, params


def create_legend(params):
    legend = []
    for j, exp in enumerate(EXPS):
        if exp['type'] == 'BL':
            txt = '{}, P: {}'.format(exp['type'], exp['params'])
        else:
            txt = '{}-{}, K: {}, N: {}, P: {} E: {}'.format(exp['ups'].name,
                                                            exp['A'].name,
                                                            exp['K'],
                                                            exp['nodes'],
                                                            params[j],
                                                            exp['epochs'])
        legend.append(txt)
    return legend


if __name__ == '__main__':
    # Set random seed
    np.random.seed(SEED)
    manual_seed(SEED)

    # Signal parameters
    Signals = {}
    Signals['n_signals'] = None
    Signals['noise'] = N_P
    Signals['to_0_1'] = False
    Signals['center'] = False

    Net = {}
    Net['laf'] = nn.Sigmoid()
    Net['af'] = nn.CELU()
    Net['bn'] = True
    Net['lr'] = 0.005

    Gs, signals = utils.read_graphs(DATASET_PATH, ATTR, MIN_SIZE, MAX_SIGNALS,
                                    to_0_1=Signals['to_0_1'],
                                    center=Signals['center'],
                                    max_size=MAX_SIZE, max_smooth=MAX_SM)

    Signals['n_signals'] = len(signals)
    start_time = time.time()
    sizes, Us, As = compute_clusters(Gs)
    print('Clustering done in {} seconds.'.format((time.time()-start_time)))

    print("CPUs used:", N_CPUS)
    err = np.zeros((len(N_P), Signals['n_signals'], N_EXPS))
    node_err = np.zeros((len(N_P), Signals['n_signals'], N_EXPS))
    start_time = time.time()
    for i, n_p in enumerate(N_P):
        print('Noise:', n_p)
        results = []
        with Pool() as pool:
            for j in range(Signals['n_signals']):
                results.append(pool.apply_async(run,
                               args=[j, signals[j].x, sizes[j], Us[j],
                                     As[j], n_p, Net, Gs[j].U]))

            for j in range(Signals['n_signals']):
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
            'Gs': [],
            'dataset': DATASET_PATH,
            'Signals': Signals,
            'Net': Net,
            'node_err': node_err,
            'err': err,
            'params': params,
            'legend': legend,
            'fmts': fmts,
        }
        utils.save_results(FILE_PREF, PATH, data)

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
from graph_deep_decoder.model import Model, Inpaint
from graph_deep_decoder import utils

# Constants
N_CPUS = cpu_count()
SAVE = True
PATH = './results/inpaint/'
FILE_PREF = 'inpainting_real'

SEED = 15
DATASET_PATH = 'dataset/graphs.mat'
MAX_SIGNALS = 50
MIN_SIZE = 50
MAX_SIZE = 150
MAX_SM = 1
ATTR = 8
P_MISS = [0, .2, .4]  # [0, .1, .2, .3, .4, .5, .6, .7]

EXPS = [
        {'ups': Ups.U_MEAN, 'gamma': 0.5, 'nodes': [4, 16, 32] + [None]*3,
         'fts': [15]*5 + [1], 'K': 0, 'epochs': 1000, 'fmt': 'o-'},
        {'ups': Ups.U_MEAN, 'gamma': 0.5, 'nodes': [4, 16, 32] + [None]*3,
         'fts': [3]*5 + [1], 'K': 0,  'epochs': 1000, 'fmt': 'o--'},

        {'ups': Ups.GF, 'K': 3,  'gamma': 0.5, 'nodes': [4, 16, 32] + [None]*3,
         'fts': [50]*5 + [1], 'epochs': 1000, 'fmt': 'X--'},
        {'ups': Ups.GF, 'K': 3,  'gamma': 0.5, 'nodes': [4, 16, 32] + [None]*3,
         'fts': [15]*5 + [1], 'epochs': 1000, 'fmt': 'X-'},
        # {'ups': Ups.GF, 'K': 3,  'gamma': 0.5, 'nodes': [4, 16, 32] + [None]*3,
        #  'fts': [3]*5 + [1], 'epochs': 1000, 'fmt': 'X--'},

        {'ups': Ups.NONE, 'gamma': 0.5, 'nodes': [None]*6,
         'K': 0,  'fts': [50]*5 + [1], 'epochs': 1000, 'fmt': 'v--'},
        {'ups': Ups.NONE, 'gamma': None, 'nodes': [None]*6,
         'fts': [15]*5 + [1], 'K': 0,  'epochs': 1000, 'fmt': 'v-'},
        # {'ups': Ups.NONE, 'gamma': 0.5, 'nodes': [None]*6,
        #  'K': 0,  'fts': [3]*5 + [1], 'epochs': 1000, 'fmt': 'v--'},
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
            nodes = [G.N if n is None else n for n in exp['nodes']]
            cluster = MultiResGraphClustering(G, nodes, nodes[0],
                                              type_A=Type_A.WEI)
            sizes[i].append(cluster.sizes)
            Us[i].append(cluster.Us)
            As[i].append(cluster.As)
    return sizes, Us, As


def run(id, x, sizes, Us, As, p_miss, n_p, Net, V):
    node_err = np.zeros(N_EXPS)
    err = np.zeros(N_EXPS)
    params = np.zeros(N_EXPS)
    x_n = ds.GraphSignal.add_noise(x, n_p)

    inp_mask = ds.GraphSignal.generate_inpaint_mask(x, p_miss)

    for i, exp in enumerate(EXPS):
        # Construct model
        dec = GraphDeepDecoder(exp['fts'], sizes[i], Us[i],
                               As=As[i], ups=exp['ups'],
                               gamma=exp['gamma'], act_fn=Net['af'],
                               last_act_fn=Net['laf'],
                               batch_norm=Net['bn'], K=exp['K'])
        model = Inpaint(dec, inp_mask, learning_rate=Net['lr'],
                        epochs=exp['epochs'])
        params[i] = model.count_params()
        model.fit(x_n)
        node_err[i], err[i] = model.test(x)
        print('Graph {}-{} ({}):  Eps: {}  Node Err: {:.8f}  Err: {:.6f}'
              .format(id, i+1, params[i], exp['epochs'], node_err[i], err[i]))
    return node_err, err, params


def create_legend(params):
    legend = []
    for j, exp in enumerate(EXPS):
        txt = '{}, K: {} N: {}, P: {} E: {}'.format(exp['ups'].name, exp['K'],
                                                    exp['nodes'],
                                                    params[j], exp['epochs'])
        legend.append(txt)
    return legend


if __name__ == '__main__':
    # Set random seed
    np.random.seed(SEED)
    manual_seed(SEED)

    # Signal parameters
    Signals = {}
    Signals['n_signals'] = None
    Signals['noise'] = 0.05
    Signals['P_MISS'] = P_MISS
    Signals['to_0_1'] = False

    Net = {}
    Net['laf'] = nn.Sigmoid()
    Net['af'] = nn.ReLU()
    Net['bn'] = True
    Net['lr'] = 0.005

    Gs, signals = utils.read_graphs(DATASET_PATH, ATTR, MIN_SIZE, MAX_SIGNALS,
                                    Signals['to_0_1'], max_size=MAX_SIZE,
                                    max_smooth=MAX_SM)
    Signals['n_signals'] = len(signals)
    start_time = time.time()
    sizes, Us, As = compute_clusters(Gs)
    print('Clustering done in {} seconds.'.format((time.time()-start_time)))

    print("CPUs used:", N_CPUS)
    err = np.zeros((len(P_MISS), Signals['n_signals'], N_EXPS))
    node_err = np.zeros((len(P_MISS), Signals['n_signals'], N_EXPS))
    start_time = time.time()
    for i, p_miss in enumerate(P_MISS):
        print('Noise:', p_miss)
        results = []
        with Pool() as pool:
            for j in range(Signals['n_signals']):
                results.append(pool.apply_async(run,
                               args=[j, signals[j].x, sizes[j], Us[j],
                                     As[j], p_miss, Signals['noise'],
                                     Net, Gs[j].U]))

            for j in range(Signals['n_signals']):
                node_err[i, j, :], err[i, j, :], params = results[j].get()

        utils.print_partial_results(node_err[i, :, :], err[i, :, :], params)

    utils.print_results(node_err, err, P_MISS, params)
    print('--- {} hours ---'.format((time.time()-start_time)/3600))
    legend = create_legend(params)
    fmts = [exp['fmt'] for exp in EXPS]
    x_label = 'Percentage of missing values'
    utils.plot_results(err, P_MISS, legend=legend, fmts=fmts, x_label=x_label)
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

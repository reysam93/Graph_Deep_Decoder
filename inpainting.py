import datetime
import os
import sys
import time
from multiprocessing import Pool, cpu_count

import numpy as np
import torch.nn as nn
from torch import manual_seed

sys.path.insert(0, 'graph_deep_decoder')
from graph_deep_decoder import datasets as ds
from graph_deep_decoder.graph_clustering import Ups, MultiResGraphClustering
from graph_deep_decoder.architecture import GraphDeepDecoder
from graph_deep_decoder.model import Inpaint
from graph_deep_decoder import utils

# Constants
N_CPUS = 1  # cpu_count()
SAVE = False
PATH = './results/inpaint/'
FILE_PREF = 'inpaint_'
SEED = 15

P_MISS = [.1, .3]  # [0, .1, .2, .3, .4, .5]
EXPS = [
    {'ups': Ups.WEI, 'gamma': 0.5, 'nodes': [4, 16, 32] + [256]*3,
     'fts': [15]*5 + [1], 'epochs': 250, 'fmt': 'x-'},
    {'ups': Ups.WEI, 'gamma': 0.5, 'nodes': [4, 16, 32] + [256]*3,
     'fts': [15]*5 + [1], 'epochs': 2500, 'fmt': 'x--'},
    {'ups': Ups.REG, 'gamma': None, 'nodes': [4, 16, 32] + [256]*3,
     'fts': [15]*5 + [1], 'epochs': 250, 'fmt': '^-'},
    {'ups': Ups.NONE, 'gamma': None, 'nodes': [256]*6, 'fts': [15]*5 + [1],
     'epochs': 250, 'fmt': 'v-'},
    {'ups': Ups.NONE, 'gamma': None, 'nodes': [256]*6, 'fts': [15]*5 + [1],
     'epochs': 2500, 'fmt': 'v--'},
    # Originals
    {'ups': Ups.WEI, 'gamma': 0.5, 'nodes': [4, 16, 32] + [256]*3,
     'fts': [3]*5 + [1], 'epochs': 250, 'fmt': 'o-'},
    {'ups': Ups.WEI, 'gamma': 0.5, 'nodes': [4, 16, 32] + [256]*3,
     'fts': [3]*5 + [1], 'epochs': 2500, 'fmt': 'o--'},
]
N_EXPS = len(EXPS)


def compute_clusters(G, root_clusts, ups):
    nodes = []
    clusters = []
    for exp in EXPS:
        if exp['nodes'] in nodes:
            i = nodes.index(exp['nodes'])
            cluster = clusters[i]
        else:
            cluster = MultiResGraphClustering(G, exp['nodes'], root_clusts,
                                              up_method=ups)
        nodes.append(exp['nodes'])
        clusters.append(cluster)
    return clusters


def run(id, Gs, Signals, Net, p_miss):
    G = ds.create_graph(Gs, SEED)
    clts = compute_clusters(G, Gs['k'], Net['ups'])
    signal = ds.GraphSignal.create(Signals['type'], G, Signals['non_lin'],
                                   Signals['L'], Signals['deltas'],
                                   to_0_1=Signals['to_0_1'])
    x_n = ds.GraphSignal.add_noise(signal.x, Signals['noise'])
    inp_mask = ds.GraphSignal.generate_inpaint_mask(signal.x, p_miss)
    mask_err = np.sum(np.square(signal.x-x_n*inp_mask))

    err = np.zeros(N_EXPS)
    node_err = np.zeros(N_EXPS)
    params = np.zeros(N_EXPS)
    for i in range(N_EXPS):
        dec = GraphDeepDecoder(EXPS[i]['fts'], clts[i].sizes, clts[i].Us,
                               As=clts[i].As, act_fn=Net['af'],
                               ups=EXPS[i]['ups'], gamma=EXPS[i]['gamma'],
                               batch_norm=Net['bn'], last_act_fn=Net['laf'])
        model = Inpaint(dec, inp_mask, learning_rate=Net['lr'],
                        epochs=EXPS[i]['epochs'])
        model.fit(x_n)
        node_err[i], err[i] = model.test(signal.x)
        params[i] = model.count_params()
        print('Graph {}-{} ({}):\tEpochs: {}\tNode Err: {:.8f}\tErr: {:.6f}'
              .format(id, i+1, params[i], EXPS[i]['epochs'], node_err[i],
                      err[i]))
    return node_err, err, mask_err, params


def create_legend(params):
    legend = []
    for i, exp in enumerate(EXPS):
        legend.append('N: {}, U: {}, P: {}, E: {}'
                      .format(exp['nodes'], exp['ups'].name, params[i],
                              exp['epochs']))
    return legend


if __name__ == '__main__':
    # Set random seed
    np.random.seed(SEED)
    manual_seed(SEED)

    # Graph parameters
    Gs = {}
    Gs['type'] = ds.SBM  # SBM or ER
    Gs['n_graphs'] = 1  # 50
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
    Signals['type'] = ds.SigType.DS  # ds.SigType.DS
    Signals['non_lin'] = ds.NonLin.MEDIAN  # ds.NonLin.MEDIAN
    Signals['deltas'] = Gs['k']
    Signals['L'] = 6
    Signals['noise'] = 0.2
    Signals['P_MISS'] = P_MISS
    Signals['to_0_1'] = False

    # Net
    Net = {}
    Net['ups'] = Ups.WEI
    Net['laf'] = nn.Tanh()
    Net['af'] = nn.Tanh()  # nn.CELU()
    Net['bn'] = True
    Net['lr'] = 0.005

    print("CPUs used:", N_CPUS)
    start_time = time.time()
    err = np.zeros((len(P_MISS), Gs['n_graphs'], N_EXPS))
    mask_err = np.zeros((len(P_MISS), Gs['n_graphs']))
    node_err = np.zeros((len(P_MISS), Gs['n_graphs'], N_EXPS))
    for i, p_miss in enumerate(P_MISS):
        print('P Miss:', p_miss)
        results = []
        with Pool(processes=N_CPUS) as pool:
            for j in range(Gs['n_graphs']):
                results.append(pool.apply_async(run,
                               args=[j, Gs, Signals, Net, p_miss]))
            for j in range(Gs['n_graphs']):
                node_err[i, j, :], err[i, j, :], mask_err[i, j], params = \
                    results[j].get()

        utils.print_partial_results(node_err[i, :, :], err[i, :, :], params)

    utils.print_results(node_err, err, P_MISS, params)
    print('--- {} hours ---'.format((time.time()-start_time)/3600))
    print('Mask Err:', np.median(mask_err, axis=1))

    legend = create_legend(params)
    fmts = [exp['fmt'] for exp in EXPS]
    utils.plot_results(err, P_MISS, legend=legend, fmts=fmts)
    if SAVE:
        data = {
            'seed': SEED,
            'exps': EXPS,
            'Gs': Gs,
            'Signals': Signals,
            'Net': Net,
            'node_err': node_err,
            'err': err,
            'mask_err': mask_err,
            'params': params,
            'legend': legend,
            'fmts': fmts,
        }
        utils.save_results(FILE_PREF, PATH, data)

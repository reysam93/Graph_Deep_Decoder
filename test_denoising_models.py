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
from graph_deep_decoder.model import Model, MeanModel, TVModel, LRModel
from graph_deep_decoder import utils

# Constants
N_CPUS = cpu_count()
SAVE = True
PATH = './results/models/'
FILE_PREF = 'models_'

SEED = 15
N_P = [0, .1, .2, .3, .4, .5]
EXPS = [
        # [15] fts para DW-MED y [6] fts pra DW-SQ2
        {'type': 'GDD', 'ups': Ups.U_MEAN, 'gamma': 0.5, 'K': None,
         'nodes': [4, 16, 32] + [256]*3, 'fts': [6]*5 + [1], 'A': Type_A.WEI,
         'fmt': 'o-'},
        {'type': 'GDD', 'ups': Ups.GF, 'gamma': None, 'K': 3,
         'nodes': [4, 16, 32] + [256]*3, 'fts': [6]*5 + [1], 'A': Type_A.WEI,
         'fmt': 'P-'},

        # {'type': 'TV', 'alpha': 1, 'fmt': 'v--'},  # mejor en DW-MED
        {'type': 'TV', 'alpha': 10, 'fmt': 'v--'},  # mejor en DW-SQ2
        {'type': 'LR', 'alpha': 0.1, 'fmt': '^--'},  # mejor en DW-MED and DW-SQ2
        {'type': 'Mean', 'fmt': 'X:'}
       ]
N_EXPS = len(EXPS)


def compute_clusters(G, root_clusts):
    clusters = []
    for exp in EXPS:
        if exp['type'] is not "GDD":
            clusters.append([])
            continue
        cluster = MultiResGraphClustering(G, exp['nodes'], root_clusts,
                                          type_A=exp['A'])
        clusters.append(cluster)
    return clusters


def create_model(exp, G, clt, Net):
    if exp['type'] is 'GDD':
        dec = GraphDeepDecoder(exp['fts'], clt.sizes, clt.Us,
                               As=clt.As, act_fn=Net['af'], ups=exp['ups'],
                               gamma=exp['gamma'], batch_norm=Net['bn'],
                               last_act_fn=Net['laf'], K=exp['K'])
        return Model(dec, learning_rate=Net['lr'], epochs=Net['epochs'])
    elif exp['type'] is 'TV':
        return TVModel(G.W.toarray(), exp['alpha'])
    elif exp['type'] is 'LR':
        return LRModel(G.L.toarray(), exp['alpha'])
    elif exp['type'] is 'Mean':
        return MeanModel(G.W.toarray())


def run(id, Gs, Signals, Net, n_p):
    G = ds.create_graph(Gs, SEED)
    clts = compute_clusters(G, Gs['k'])
    signal = ds.GraphSignal.create(Signals['type'], G, Signals['non_lin'],
                                   Signals['L'], Signals['deltas'])
    x_n = ds.GraphSignal.add_noise(signal.x, n_p)

    err = np.zeros(N_EXPS)
    node_err = np.zeros(N_EXPS)
    params = np.zeros(N_EXPS)
    for i, exp in enumerate(EXPS):
        model = create_model(exp, G, clts[i], Net)
        model.fit(x_n)
        node_err[i], err[i] = model.test(signal.x)
        params[i] = model.count_params()
        print('Graph {}-{}:\tNode Err: {:.8f}\tErr: {:.6f}'
              .format(id, i+1, node_err[i], err[i]))
    return node_err, err, params


def create_legend(params):
    legend = []
    for i, exp in enumerate(EXPS):
        txt = exp['type'] + ', '
        if exp['type'] is 'GDD':
            txt += 'Ups: {}, G: {}, K: {}, P: {}'.format(exp['ups'].name,
                                                         exp['gamma'],
                                                         exp['K'], params[i])
        elif exp['type'] in ['TV', 'LR']:
            txt += 'Alpha: {}'.format(exp['alpha'])
        legend.append(txt)
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
    Signals['n_signals'] = 1  # 50
    Signals['type'] = ds.SigType.DW
    Signals['non_lin'] = ds.NonLin.SQ2
    Signals['deltas'] = Gs['k']
    Signals['L'] = 6
    Signals['noise'] = N_P
    Signals['missing'] = 0

    Net = {}
    Net['laf'] = nn.Tanh()
    Net['af'] = nn.Tanh()  # nn.CELU()
    Net['bn'] = True
    Net['lr'] = 0.005
    Net['epochs'] = 2500

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

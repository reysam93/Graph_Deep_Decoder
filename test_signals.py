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
from graph_deep_decoder.graph_clustering import Type_A, MultiResGraphClustering
from graph_deep_decoder.architecture import GraphDeepDecoder, Ups
from graph_deep_decoder.model import Model, BLModel
from graph_deep_decoder import utils


# Constants
N_CPUS = cpu_count()
SAVE = True
PATH = './results/input/'
FILE_PREF = 'input_'

SEED = 15
N_P = [0, .1, .2, .3, .4, .5]

SIGS = [{'type': ds.SigType.DW, 'non_lin': ds.NonLin.MEDIAN, 'fmt': '-'},
        {'type': ds.SigType.DW, 'non_lin': ds.NonLin.SQ2, 'fmt': '--'},
        # {'type': ds.SigType.NOISE, 'non_lin': ds.NonLin.NONE, 'fmt': '-.'},
        ]

EXPS = [{'type': 'DD', 'A': Type_A.WEI, 'ups': Ups.U_MEAN, 'fts': [15]*5 + [1],
         'nodes': [4, 16, 32] + [256]*3, 'K': 0, 'epochs': 500, 'fmt': 'o'},
        {'type': 'DD', 'A': Type_A.WEI,  'ups': Ups.GF, 'fts': [15]*5 + [1],
         'nodes': [4, 16, 32] + [256]*3, 'K': 3, 'epochs': 500, 'fmt': 'P'},
        {'type': 'DD', 'A': Type_A.NONE, 'ups': Ups.NONE, 'fts': [15]*5 + [1],
         'nodes': [256]*6, 'epochs': 500, 'K': 0, 'fmt': 'v'},

        {'type': 'DD', 'A': Type_A.WEI, 'ups': Ups.U_MEAN, 'fts': [6]*5 + [1],
         'nodes': [4, 16, 32] + [256]*3, 'K': 0, 'epochs': 500, 'fmt': 'X'},
        {'type': 'DD', 'A': Type_A.WEI,  'ups': Ups.GF, 'fts': [6]*5 + [1],
         'nodes': [4, 16, 32] + [256]*3, 'K': 3, 'epochs': 500, 'fmt': '>'},
        {'type': 'DD', 'A': Type_A.NONE, 'ups': Ups.NONE, 'fts': [6]*5 + [1],
         'nodes': [256]*6, 'epochs': 500, 'K': 0, 'fmt': '^'}]

# EXPERIMENTO ORIGINAL!
# SIGS = [{'type': ds.SigType.DS, 'non_lin': ds.NonLin.NONE, 'fmt': '-'},
#         {'type': ds.SigType.DS, 'non_lin': ds.NonLin.MEDIAN, 'fmt': '--'}, ]
#         # {'type': ds.SigType.DW, 'non_lin': ds.NonLin.SQUARE, 'fmt': '--'},
#         # {'type': ds.SigType.SM, 'non_lin': ds.NonLin.MEDIAN, 'fmt': ':'},
#         # {'type': ds.SigType.NOISE, 'non_lin': ds.NonLin.NONE, 'fmt': '-.'}]

# EXPS = [{'type': 'BL', 'params': 63, 'epochs': 0, 'fmt': 'o'},
#         {'type': 'DD', 'ups': Ups.WEI, 'fts': [3]*5 + [1],
#          'nodes': [4, 16, 32] + [256]*3, 'epochs': 300, 'fmt': 'X'},
#         # {'type': 'DD', 'ups': Ups.REG, 'fts': [15]*3 + [1],
#         #  'nodes': [4] + [256]*3, 'epochs': 3000, 'fmt': '^'},

#         {'type': 'DD', 'ups': Ups.WEI, 'fts': [15]*5 + [1],
#          'nodes': [4, 16, 32] + [256]*3, 'epochs': 250, 'fmt': 'P'},
#         {'type': 'DD', 'ups': Ups.WEI, 'fts': [15]*3 + [1],
#          'nodes': [4] + [256]*3, 'epochs': 250, 'fmt': 'v'}]

N_EXPS = len(SIGS)*len(EXPS)


def compute_clusters(G, root_clusts):
    clusters = []
    for exp in EXPS:
        if exp['type'] == 'BL':
            clusters.append(None)
            continue

        cluster = MultiResGraphClustering(G, exp['nodes'], root_clusts,
                                          type_A=exp['A'])
        clusters.append(cluster)
    return clusters


def run(id, Gs, Signals, Net, n_p):
    G = ds.create_graph(Gs, SEED)
    clts = compute_clusters(G, Gs['k'])

    node_err = np.zeros(N_EXPS)
    err = np.zeros(N_EXPS)
    params = np.zeros(N_EXPS)
    for i, sig in enumerate(SIGS):
        signal = ds.GraphSignal.create(sig['type'], G, sig['non_lin'],
                                       Signals['L'], Signals['deltas'],
                                       to_0_1=Signals['to_0_1'])
        x_n = ds.GraphSignal.add_noise(signal.x, n_p)
        for j, exp in enumerate(EXPS):
            cont = i*len(EXPS)+j

            # Construct model
            if exp['type'] == 'BL':
                model = BLModel(G.U, exp['params'])
            else:
                dec = GraphDeepDecoder(exp['fts'], clts[j].sizes, clts[j].Us,
                                       As=clts[j].As, ups=exp['ups'],
                                       act_fn=Net['af'], K=exp['K'],
                                       last_act_fn=Net['laf'],
                                       batch_norm=Net['bn'])
                model = Model(dec, learning_rate=Net['lr'],
                              epochs=exp['epochs'])

            params[cont] = model.count_params()
            model.fit(x_n)
            node_err[cont], err[cont] = model.test(signal.x)
            print('Signal {}-{} ({}):  Eps: {}  Node Err: {:.8f}  Err: {:.6f}'
                  .format(id, cont+1, params[cont], exp['epochs'],
                          node_err[cont], err[cont]))
    return node_err, err, params


def create_legend(params):
    legend = []
    for sig in SIGS:
        sig_txt = 'S: {}-{}, '.format(sig['type'].name, sig['non_lin'].name)
        for j, exp in enumerate(EXPS):
            if exp['type'] == 'BL':
                txt = sig_txt + '{}, P: {}'.format(exp['type'], exp['params'])
            else:
                txt = sig_txt + '{}, A: {}, P: {}'.format(exp['ups'], exp['A'],
                                                          params[j])
            legend.append(txt)
    return legend


def create_fmts():
    fmts = []
    for sig in SIGS:
        for exp in EXPS:
            fmts.append(exp['fmt'] + sig['fmt'])
    return fmts


if __name__ == '__main__':
    # Set random seed
    np.random.seed(SEED)
    manual_seed(SEED)

    # Graph parameters
    Gs = {}
    Gs['type'] = ds.SBM  # SBM, ER or BA
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
    Signals['n_signals'] = 1
    Signals['deltas'] = deltas = Gs['k']
    Signals['L'] = L = 6
    Signals['noise'] = N_P
    Signals['type'] = SIGS
    Signals['to_0_1'] = False

    # Network (architecture + model) parameters
    Net = {}
    Net['laf'] = nn.Tanh()
    Net['af'] = nn.Tanh()  # nn.CELU()
    Net['bn'] = True
    Net['lr'] = 0.005
    Net['dr'] = 1

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
    fmts = create_fmts()
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

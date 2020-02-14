import datetime
import os
import sys
import time
from multiprocessing import Pool, cpu_count

import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
from scipy.io import loadmat
from torch import manual_seed

from pygsp import plotting

sys.path.insert(0, 'graph_deep_decoder')
from graph_deep_decoder import datasets as ds
from graph_deep_decoder.graph_clustering import Type_A, MultiResGraphClustering
from graph_deep_decoder.architecture import GraphDeepDecoder, Ups
from graph_deep_decoder.model import Model, Inpaint
from graph_deep_decoder import utils


def get_graph_params(g_type):
    Gs = {}
    Gs['type'] = g_type
    Gs['N'] = 256
    if g_type is ds.SBM:
        Gs['k'] = 4
        Gs['p'] = 0.25
        Gs['q'] = [[0, 0.0075, 0, 0.0],
                   [0.0075, 0, 0.004, 0.0025],
                   [0, 0.004, 0, 0.005],
                   [0, 0.0025, 0.005, 0]]
        Gs['type_z'] = ds.RAND
    elif g_type is ds.ER:
        Gs['p'] = 0.05
    elif g_type is ds.BA:
        Gs['m0'] = 2
        Gs['m'] = 2
    return Gs


def get_energy():
    pass


def print_sumary(energy_L, energy_A, error):
    print('Mean energy L:', np.mean(energy_L))
    print('Mean energy A:', np.mean(energy_A))
    print('Mean Error:', np.mean(error))


def plot_coefs_hist(coefs):
    size = [coefs.shape[0]*coefs.shape[1], coefs.shape[2]]
    coefs = coefs.reshape(size)
    for i in range(coefs.shape[1]):
        plt.figure()
        plt.hist(coefs[:, i])
        plt.title('K={}'.format(i))
    plt.figure()
    array_coefs = coefs.reshape([coefs.shape[0]*coefs.shape[1]])
    plt.hist(coefs)
    plt.title('All coefficients')
    plt.show()


if __name__ == '__main__':
    seed = None
    # Set random seed
    # seed = 15
    # np.random.seed(seed)
    # manual_seed(seed)

    g_type = ds.SBM
    n_graphs = 50
    Gs = get_graph_params(g_type)

    # Signal parameters
    Signals = {}
    Signals['type'] = ds.SigType.DW
    Signals['non_lin'] = ds.NonLin.MEDIAN
    Signals['deltas'] = Gs['k']
    Signals['L'] = 6
    Signals['noise'] = 0.2
    Signals['missing'] = 0

    # Network parameters (Cluster + Decoder + Model)
    Net = {}
    Net['laf'] = nn.Tanh()
    Net['af'] = nn.Tanh()
    Net['bn'] = True
    Net['lr'] = 0.005
    Net['epochs'] = 2000
    Net['fts'] = [15]*5 + [1]
    Net['ups'] = Ups.GF
    Net['gamma'] = 0.5
    Net['K'] = 4
    Net['nodes'] = [4, 16, 32] + [256]*3
    Net['root_clust'] = Gs['k']
    Net['type_A'] = Type_A.WEI

    # Multiple CPUS??
    k_first = int(Gs['N']*0.25)
    n_ups = np.unique(Net['nodes']).size-1
    energy_L = np.zeros(n_graphs)
    energy_A = np.zeros(n_graphs)
    coefs = np.zeros((n_graphs, n_ups, Net['K']))
    err = np.zeros(n_graphs)
    for i in range(n_graphs):
        G = ds.create_graph(Gs, seed)

        # Energy from L
        energy_highest = np.linalg.norm(G.e[G.N-k_first-1:-1])
        energy_L[i] = (energy_highest/np.linalg.norm(G.e))**2

        # Energy from A
        eigenvals, _ = np.linalg.eig(G.W.todense())
        energy_highest = np.linalg.norm(eigenvals[:k_first])
        energy_A[i] = (energy_highest/np.linalg.norm(eigenvals))**2

        clt = MultiResGraphClustering(G, Net['nodes'], Net['root_clust'],
                                      type_A=Net['type_A'])
        signal = ds.GraphSignal.create(Signals['type'], G,
                                       Signals['non_lin'],
                                       Signals['L'], Signals['deltas'])
        x_n = ds.GraphSignal.add_noise(signal.x, Signals['noise'])

        dec = GraphDeepDecoder(Net['fts'], clt.sizes, clt.Us, As=clt.As,
                               act_fn=Net['af'], ups=Net['ups'], K=Net['K'],
                               gamma=Net['gamma'], batch_norm=Net['bn'],
                               last_act_fn=Net['laf'])
        model = Model(dec, learning_rate=Net['lr'], epochs=Net['epochs'])
        model.fit(x_n)
        _, err[i] = model.test(signal.x)
        coefs[i, :, :] = model.get_filter_coefs()
        print('G {}: E_L: {} E_A: {} Err: {}'.format(i, energy_L[i],
                                                     energy_A[i], err[i]))

    print_sumary(energy_L, energy_A, err)
    # Mean over realizations
    print('Mean coefficients (separated by layers):')
    print(np.mean(coefs, axis=0))
    # Mean over realizations and layers
    print('Mean coefficients (including layers):')
    print(np.mean(coefs, axis=(0, 1)))
    # Mean over realizations and taps
    print('Mean coefficients (including all taps):')
    print(np.mean(coefs, axis=(0, 1)))
    plot_coefs_hist(coefs)

    

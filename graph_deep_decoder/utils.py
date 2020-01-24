import os
import datetime

import matplotlib.pyplot as plt
from matplotlib import rc
import numpy as np
from pygsp.graphs import ErdosRenyi, Graph, StochasticBlockModel
from scipy import sparse
from scipy.cluster.hierarchy import dendrogram, fcluster, linkage
from scipy.sparse.csgraph import dijkstra
from scipy.io import loadmat
from sklearn.cluster import AgglomerativeClustering

from graph_deep_decoder import datasets as ds


def plot_overfitting(err, err_val, fmts, legend, show=True):
    rc('text', usetex=True)
    fig, ax = plt.subplots()
    # ax.semilogy(err, label='Train Err')
    # ax.semilogy(err_val, label='Val Err')
    # ax.legend()

    for i in range(err.shape[1]):
        label_train = 'Train: $||x_n-x_{est}||$ '
        label_val = 'Val: $||x_0-x_{est}||$ '
        ax.semilogy(err[:, i], fmts[i]+'-', label=label_train + legend[i])
        ax.semilogy(err_val[:, i], fmts[i]+'--', label=label_val + legend[i])

    ax.legend()
    plt.grid(True, which='both')
    plt.tight_layout()
    if show:
        plt.show()


def print_partial_results(node_err, err, params):
    node_err = np.median(node_err, 0)
    med_err = np.median(err, 0)
    std_err = np.std(err, 0)
    if params is None:
        params = np.zeros(med_err.size)
    for i in range(med_err.size):
        print('\t{} ({} params):\tNode MSE: {:.8}\tMedian MSE: {:.6f}\tSTD: {:.6f}'
              .format(i+1, params[i], node_err[i], med_err[i], std_err[i]))


def print_results(node_err, err, noises, params=None):
    for i, noise in enumerate(noises):
        print('Noise: ', noise)
        print_partial_results(node_err[i, :, :], err[i, :, :], params)


def save_results(file_pref, path, data, verbose=True):
    if not os.path.isdir(path):
        os.makedirs(path)
    timestamp = datetime.datetime.now().strftime("%Y_%m_%d-%H_%M")
    if path[-1] != '/':
        path += '/'
    path = path + file_pref + timestamp
    np.save(path, data)
    if verbose:
        print('SAVED as:', os.getcwd(), path)


def plot_results(err, x_axis, legend=None, fmts=None, x_label=None):
    median_err = np.median(err, axis=1)
    fig, ax = plt.subplots()
    for i in range(median_err.shape[1]):
        if fmts is None:
            plt.semilogy(x_axis, median_err[:, i])
        else:
            plt.semilogy(x_axis, median_err[:, i], fmts[i])

    if x_label is None:
        x_label = 'Normalized Noise Power'
    plt.xlabel(x_label, fontsize=16)
    plt.ylabel('Median Error', fontsize=16)
    plt.gca().autoscale(enable=True, axis='x', tight=True)
    plt.grid(True, which='both')
    if legend is not None:
        plt.legend(legend)
        # plt.legend(legend, prop={'size': 14})
    plt.show()


def remove_indexes(data, skip_indexes):
    data['err'] = np.delete(data['err'], skip_indexes, axis=2)
    skip_indexes.reverse()
    for i in skip_indexes:
        del data['legend'][i]
        del data['fmts'][i]
    return data


def plot_from_file(file, skip=[]):
    data = np.load(file).item()

    if skip:
        data = remove_indexes(data, skip)

    err = data['err']
    noise = data['Signals']['noise']
    legend = data['legend']
    fmts = data['fmts']
    plot_results(err, noise, legend, fmts)


def print_sumary(data):
    print('Graph parameters:')
    print(data['Gs'])
    print('Signals parameters:')
    print(data['Signals'])
    print('Network parameters:')
    print(data['Net'])
    print('Experiments:')
    print(data['exps'])


def print_from_file(file):
    data = np.load(file).item()
    err = data['err']
    node_err = data['node_err']
    noise = data['Signals']['noise']
    params = data['params']
    print_sumary(data)
    print_results(node_err, err, noise, params)


def read_graphs(dataset_path, attr, min_size=50, max_signals=100,
                to_0_1=False, center=False):
    signals = []
    Gs = []
    g_sizes = []
    gs_mat = loadmat(dataset_path)

    for i, A in enumerate(gs_mat['cell_A']):
        if len(signals) >= max_signals:
            break
        G = Graph(A[0])
        if G.N < min_size or not G.is_connected():
            continue

        if gs_mat['cell_X'][0][0].shape[1] == 1:
            signal = ds.DeterministicGS(G, gs_mat['cell_X'][i][0][:, 0])
        else:
            signal = ds.DeterministicGS(G, gs_mat['cell_X'][i][0][:, attr-1])

        if np.linalg.norm(signal.x) == 0:
            continue

        if center:
            signal.center()
        if to_0_1:
            signal.to_0_1_interval()
        signal.to_unit_norm()

        G.compute_fourier_basis()
        G.set_coordinates('spring')
        Gs.append(G)
        g_sizes.append(G.N)
        signals.append(signal)

    print('Graphs read:', len(Gs), 'from:', i, 'mean size:', np.mean(g_sizes))
    return Gs, signals

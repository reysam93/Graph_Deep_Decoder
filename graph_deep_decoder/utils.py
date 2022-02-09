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
from graph_deep_decoder.architecture import GraphDecoder


def plot_overfitting(err, fmts, legend, show=True):
    fig, ax = plt.subplots()
    # ax.semilogy(err, label='Train Err')
    # ax.semilogy(err_val, label='Val Err')
    # ax.legend()

    for i in range(err.shape[1]):
        ax.semilogy(err[:, i], fmts[i]+'-', label=legend[i])

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

    if isinstance(noise, list):
        plot_results(err, noise, legend, fmts)
    else:
        p_miss = data['Signals']['P_MISS']
        x_label = 'Percentage of missing values'
        plot_results(err, p_miss, legend, fmts, x_label=x_label)


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
    if isinstance(noise, list):
        print_results(node_err, err, noise, params)
    else:
        p_miss = data['Signals']['P_MISS']
        print_results(node_err, err, p_miss, params)


def read_graphs(dataset_path, attr, min_size=50, max_signals=100,
                to_0_1=False, center=False, max_size=None, max_smooth=None,
                max_bl_err=None):
    signals = []
    Gs = []
    g_sizes = []
    gs_mat = loadmat(dataset_path)

    for i, A in enumerate(gs_mat['cell_A']):
        if len(signals) >= max_signals:
            break
        G = Graph(A[0])
        G.compute_fourier_basis()
        if G.N < min_size or not G.is_connected():
            continue
        if max_size and G.N > max_size:
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

        if max_smooth and signal.smoothness() > max_smooth:
            continue

        if max_bl_err and signal.check_bl_err(coefs=0.25, firsts=True) > max_bl_err:
            continue

        G.compute_fourier_basis()
        G.set_coordinates('spring')
        Gs.append(G)
        g_sizes.append(G.N)
        signals.append(signal)

    print('Graphs read:', len(Gs), 'from:', i, 'mean size:', np.mean(g_sizes))
    return Gs, signals


def ordered_eig(M):
    """
    Ensure the eigendecomposition of M is ordered
    """
    eig_val, eig_vec = np.linalg.eig(M)
    idx = np.flip(np.argsort(np.abs(eig_val)), axis=0)
    eig_val = np.real(eig_val[idx])
    eig_vec = np.real(eig_vec[:, idx])
    return eig_val, eig_vec


def choose_eig_sign(EigA, EigB):
    """
    Ensure that most of the signs between eigA and eigB are the same
    by multiplying eigA by -1 if needed.
    """
    diff_signs = np.sum(np.sign(EigA) != np.sign(EigB), axis=0)
    mask = np.where(diff_signs > EigA.shape[0]/2, -1, 1)
    return mask*EigA


def create_filter(S, ps, x=None):
    if ps['type'] is 'BLH':
        _, V = ordered_eig(S)
        V = np.real(V)
        eigvalues = np.ones(V.shape[0])*0.001
        bl_k = int(S.shape[0]*ps['k'])
        if ps['firsts']:
            eigvalues[:bl_k] = 1
        else:
            x_freq = V.T.dot(x)
            idx = np.flip(np.abs(x_freq).argsort(), axis=0)[:bl_k]
            eigvalues[idx] = 1

        H = V.dot(np.diag(eigvalues).dot(V.T))
    elif ps['type'] is 'RandH':
        hs = np.random.rand(ps['K'])
        hs /= np.sum(hs)
    elif ps['type'] is 'FixedH':
        hs = ps['hs']
    else:
        print('Unkwown filter type')
        return None

    if ps['type'] is not 'BLH':
        H = np.zeros((S.shape))
        for l, h in enumerate(hs):
            H += h*np.linalg.matrix_power(S, l)

    if ps['H_norm']:
        H /= np.linalg.norm(H)

    return H
    
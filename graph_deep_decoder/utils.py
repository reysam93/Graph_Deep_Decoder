import os
import datetime

import matplotlib.pyplot as plt
import numpy as np
from pygsp.graphs import ErdosRenyi, Graph, StochasticBlockModel
from scipy import sparse
from scipy.cluster.hierarchy import dendrogram, fcluster, linkage
from scipy.sparse.csgraph import dijkstra
from sklearn.cluster import AgglomerativeClustering


def plot_overfitting(err, err_val, show=True):
    fig, ax = plt.subplots()
    ax.semilogy(err, label='Train Err')
    ax.semilogy(err_val, label='Val Err')
    ax.legend()
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
        print('SAVED as:', path)


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


def plot_from_file(file):
    data = np.load(file).item()
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

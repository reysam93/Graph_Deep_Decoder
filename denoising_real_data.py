import os, sys
import time, datetime
from graph_deep_decoder import utils
from graph_deep_decoder.architecture import GraphDeepDecoder
from multiprocessing import Pool 
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
from scipy.io import loadmat 
from pygsp.graphs import Graph

SEED = 15
DATASET_PATH = 'dataset/graphs.mat'
MAX_SIGNALS = 100
MIN_SIZE = 75
ATTR = 6
N_P = [0, .05, .1, .2, .3, .5]
EXPERIMENTS = [{'ups': 'original', 'arch': [3,3,3], 't': [4,16,64,None], 'gamma': None},
               {'ups': 'no_A', 'arch': [3,3,3], 't': [4,16,64,None], 'gamma': None},
               {'ups': 'weighted', 'arch': [3,3,3], 't': [4,16,64,None], 'gamma': 0.5},
               {'ups': 'original', 'arch': [3,3,3], 't': [2,8,32,None], 'gamma': None},
               {'ups': 'no_A', 'arch': [3,3,3], 't': [2,8,32,None], 'gamma': None},
               {'ups': 'weighted', 'arch': [3,3,3], 't': [2,8,32,None], 'gamma': 0.5},
               {'ups': 'original', 'arch': [4,4,4,4], 't': [4,16,32,64,None], 'gamma': None},
               {'ups': 'no_A', 'arch': [4,4,4,4], 't': [4,16,32,64,None], 'gamma': None},
               {'ups': 'weighted', 'arch': [4,4,4,4], 't': [4,16,32,64,None], 'gamma': 0.5},
               {'ups': 'original', 'arch': [4,4,4,4], 't': [2,8,16,32,None], 'gamma': None},
               {'ups': 'no_A', 'arch': [4,4,4,4], 't': [2,8,16,32,None], 'gamma': None},
               {'ups': 'weighted', 'arch': [4,4,4,4], 't': [2,8,16,32,None], 'gamma': 0.5},]
N_EXPS = len(EXPERIMENTS)

FMTS = ['o-', '^-', '+-', 'x-', 'o--', '^--', '+--', 'x--', 'o:', '^:', '+:', 'x:']


def read_graphs():
    signals = []
    Gs = []
    graphs_mat = loadmat(DATASET_PATH)
    for i, A in  enumerate(graphs_mat['cell_A']):
        if len(signals) >= MAX_SIGNALS:
            break
        G = Graph(A[0])
        if G.N < MIN_SIZE or not G.is_connected():
            continue
        
        signal = utils.DeterministicGS(G, graphs_mat['cell_X'][i][0][:,ATTR])
        signal.signal_to_0_1_interval()
        signal.to_unit_norm()
        Gs.append(G)
        signals.append(signal)
        
    print('Graphs readed:', len(Gs))
    return Gs, signals

def compute_clusters(Gs):
    sizes = []
    descendances = []
    hier_As = []

    for i, G in enumerate(Gs):
        sizes.append([])
        descendances.append([])
        hier_As.append([])
        for exp in EXPERIMENTS:
            exp['t'][-1] = G.N
            root_clust = exp['arch'][0]
            cluster = utils.MultiRessGraphClustering(G, exp['t'], root_clust)
            sizes[i].append(cluster.clusters_size)
            descendances[i].append(cluster.compute_hierarchy_descendance())
            hier_As[i].append(cluster.compute_hierarchy_A(exp['ups']))
    return sizes, descendances, hier_As

def denoise_real(id, x, sizes, descendances, hier_As, n_p):
    error = np.zeros(N_EXPS)
    params = np.zeros(N_EXPS)
    x_n = utils.RandomGraphSignal.add_noise(x, n_p)
    for i, exp in enumerate(EXPERIMENTS):
        dec = GraphDeepDecoder(descendances[i], hier_As[i], sizes[i],
                        n_channels=exp['arch'], upsampling=exp['ups'],
                        gamma=exp['gamma'], last_act_fun=nn.Sigmoid())
        dec.build_network()
        x_est, _ = dec.fit(x_n)
        error[i] = np.sum(np.square(x-x_est))/np.linalg.norm(x)
        params[i] = dec.count_params()
        print('Signal: {} Scenario {}: Error: {:.4f}'
                            .format(id, i, error[i]))
    return error, params

def print_results(mean_mse, median_mse, params, n_p):
    print('NOISE POWER:', n_p)
    for i, exp in enumerate(EXPERIMENTS):
        print('{}. (EXP {}) '.format(i, exp))
        print('\tMean Error: {}\tMedian Error: {}\tParams: {}'
                    .format(mean_mse[i], median_mse[i], params[i]))

def plot_results(mean_mse):
    plt.figure()
    for i in range(N_EXPS):
        plt.plot(N_P, mean_mse[:,i], FMTS[i])
    
    plt.xlabel('Noise Power')
    plt.ylabel('Mean MSE')
    legend = [str(exp) for exp in EXPERIMENTS]
    plt.legend(legend)

def save_results(mse_est):
    if not os.path.isdir('./results/denoising'):
        os.makedirs('./results/denoising')
    timestamp = datetime.datetime.now().strftime("%Y_%m_%d-%H_%M")
    data = {'mse': mse_est, 'EXPERIMENTS': EXPERIMENTS, 'N_P': N_P,
            'SEED': SEED, 'FMTS': FMTS}
    path = './results/denoising/denoise_real_' + timestamp
    np.save(path, data)
    print('SAVED as:', path)

if __name__ == '__main__':
    # Set seeds
    utils.RandomGraphSignal.set_seed(SEED)
    GraphDeepDecoder.set_seed(SEED)
    
    Gs, signals = read_graphs()
    n_signals = len(signals)
    sizes, descendances, hier_As = compute_clusters(Gs)

    mse_est = np.zeros((len(N_P), n_signals, N_EXPS))
    start_time = time.time()
    for i, n_p in enumerate(N_P):  
        with Pool() as pool:
            for j in range(n_signals):
                result = pool.apply_async(denoise_real,
                        args=[j, signals[j].x, sizes[j], descendances[j],
                                hier_As[j], n_p])

            for j in range(n_signals):
                    mse_est[i,j,:], n_params = result.get()

        print_results(np.mean(mse_est[i,:,:], axis=0),
                        np.median(mse_est[i,:,:], axis=0), n_params, n_p)
    plot_results(np.mean(mse_est, axis=1))
    save_results(mse_est)
    print('--- {} minutes ---'.format((time.time()-start_time)/60))
    plt.show()
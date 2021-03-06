import os, sys
sys.path.insert(0, '../graph_deep_decoder')
import time, datetime
from graph_deep_decoder import utils
from graph_deep_decoder import graph_signals as gs
from graph_deep_decoder.architecture import GraphDeepDecoder
from multiprocessing import Pool, cpu_count
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
from scipy.io import loadmat 
from pygsp.graphs import Graph

SEED = 15
N_CPUS = cpu_count()
DATASET_PATH = 'dataset/graphs.mat'
MAX_SIGNALS = 100
MIN_SIZE = 50
ATTR = 6
N_P = [0, .1, .2, .3, .4, .5]


EXPERIMENTS = ['bandlimited',
               {'ups': 'original', 'arch': [3,3], 't': [24,48,None], 'gamma': None},
               {'ups': 'weighted', 'arch': [3,3], 't': [24,48,None], 'gamma': 0}]

# Architecture
BATCH_NORM = True
LAST_ACT_FUN = nn.Tanh()
ACT_FUN = nn.Tanh()
# Bandlimited
N_PARAMS = 48

N_EXPS = len(EXPERIMENTS)


def read_graphs():
    signals = []
    Gs = []
    graphs_mat = loadmat(DATASET_PATH)
    g_sizes = []
    for i, A in  enumerate(graphs_mat['cell_A']):
        if len(signals) >= MAX_SIGNALS:
            break
        G = Graph(A[0])
        if G.N < MIN_SIZE or not G.is_connected():
            continue
        
        signal = gs.DeterministicGS(G, graphs_mat['cell_X'][i][0][:,ATTR])
        if np.linalg.norm(signal.x) == 0:
            continue
        #signal.normalize()
        #signal.signal_to_0_1_interval()
        signal.to_unit_norm()
        G.compute_fourier_basis()
        Gs.append(G)
        g_sizes.append(G.N) 
        signals.append(signal)

    print('Graphs readed:', len(Gs), 'from:',i, 'mean size:', np.mean(g_sizes))
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
            if not isinstance(exp,dict):
                sizes[i].append(None)
                descendances[i].append(None)
                hier_As[i].append(None)
                continue

            exp['t'][-1] = G.N
            root_clust = exp['arch'][0]
            cluster = utils.MultiRessGraphClustering(G, exp['t'], root_clust)
            sizes[i].append(cluster.clusters_size)
            descendances[i].append(cluster.compute_hierarchy_descendance())
            hier_As[i].append(cluster.compute_hierarchy_A(exp['ups']))
    return sizes, descendances, hier_As

def denoise_real(id, x, sizes, descendances, hier_As, n_p, V):
    error = np.zeros(N_EXPS)
    x_n = gs.GraphSignal.add_noise(x, n_p)
    for i, exp in enumerate(EXPERIMENTS):
        if not isinstance(exp, dict):
            x_est = utils.bandlimited_model(x_n, V, n_coefs=N_PARAMS, max_coefs=False)
            n_params = N_PARAMS
        else:
            dec = GraphDeepDecoder(descendances[i], hier_As[i], sizes[i],
                            n_channels=exp['arch'], upsampling=exp['ups'],
                            gamma=exp['gamma'], last_act_fun=LAST_ACT_FUN,
                            act_fun=ACT_FUN, batch_norm=BATCH_NORM)
            dec.build_network()
            n_params = dec.count_params()
            x_est, _ = dec.fit(x_n, n_iter=4000)
        error[i] = np.sum(np.square(x-x_est))/np.square(np.linalg.norm(x))
        print('Signal: {} Scenario {}: Error: {:.4f}, Params: {}, N: {}'
                            .format(id, i, error[i], n_params, x_n.size))
    return error

def print_results(mse_est, n_p):
    mean_mse = np.mean(mse_est, axis=0)
    median_mse = np.median(mse_est, axis=0)
    std = np.std(mse_est, axis=0)
    print('NOISE POWER:', n_p)
    for i, exp in enumerate(EXPERIMENTS):
        print('{}. (EXP {}) '.format(i, exp))
        print('\tMean Error: {}\tMedian Error: {}\tSTD: {}'
                    .format(mean_mse[i], median_mse[i], std[i]))

def save_results(error):
    if not os.path.isdir('./results/denoising'):
        os.makedirs('./results/denoising')
    timestamp = datetime.datetime.now().strftime("%Y_%m_%d-%H_%M")
    data = {'error': error, 'EXPERIMENTS': EXPERIMENTS, 'N_P': N_P,
            'SEED': SEED, 'natch_norm': BATCH_NORM, 'act_fun': ACT_FUN,
            'last_act_fun': LAST_ACT_FUN}
    path = './results/denoising/denoise_real_' + timestamp
    np.save(path, data)
    print('SAVED as:', path)

if __name__ == '__main__':
    # Set seeds
    gs.GraphSignal.set_seed(SEED)
    GraphDeepDecoder.set_seed(SEED)
    
    Gs, signals = read_graphs()
    n_signals = len(signals)
    sizes, descendances, hier_As = compute_clusters(Gs)

    print("CPUs used:", N_CPUS)
    error = np.zeros((len(N_P), n_signals, N_EXPS))
    start_time = time.time()
    for i, n_p in enumerate(N_P):
        print('noise',i,':',n_p)
        results = []
        with Pool(processes=N_CPUS) as pool:
            for j in range(n_signals):
                results.append(pool.apply_async(denoise_real,
                        args=[j, signals[j].x, sizes[j], descendances[j],
                                hier_As[j], n_p, Gs[j].U]))

            for j in range(n_signals):
                error[i,j,:] = results[j].get()
        print_results(error[i,:,:], n_p)

    save_results(error)
    print('--- {} minutes ---'.format((time.time()-start_time)/60))

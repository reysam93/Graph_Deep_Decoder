import sys
import os
import time, datetime
from multiprocessing import Pool 
sys.path.insert(0, 'graph_deep_decoder')
from graph_deep_decoder import utils
from graph_deep_decoder.architecture import GraphDeepDecoder

import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn

# Constants
N_SIGNALS = 2 
SEED = 15
n_p = 0

INPUTS = ['linear', 'non-linear', 'median', 'comb']
EXPERIMENTS = [{'ups': 'original', 'arch': [3,3,3], 't': [4,16,64,256]},
               {'ups': 'no_A', 'arch': [3,3,3], 't': [4,16,64,256]},
               {'ups': 'binary', 'arch': [3,3,3], 't': [4,16,64,256]},
               {'ups': 'weighted', 'arch': [3,3,3], 't': [4,16,64,256]}]
N_EXPS = len(INPUTS)*len(EXPERIMENTS)

def compute_clusters(G, root_clust):
    sizes = []
    descendances = []
    hier_As = []
    for exp in EXPERIMENTS:
        exp['t'][-1] = G.N
        cluster = utils.MultiRessGraphClustering(G, exp['t'], root_clust)
        sizes.append(cluster.clusters_size)
        descendances.append(cluster.compute_hierarchy_descendance())
        hier_As.append(cluster.compute_hierarchy_A(exp['ups']))
    return sizes, descendances, hier_As

def create_signal(signal_type, G, L, k):
    if signal_type == 'linear':
        signal = utils.DifussedSparseGS(G,L,k)
    elif signal_type == 'non-linear':
        signal = utils.NonLinealDSGS(G,L,k)
    elif signal_type == 'median':
        signal = utils.MedianDSGS(G,L,k)
    elif signal_type == 'comb':
        signal = utils.NLCombinationsDSGS(G,L,k)
    else:
        raise RuntimeError('Unknown signal type')
    signal.signal_to_0_1_interval()
    signal.to_unit_norm()
    return signal

def test_input(id, signals, sizes, descendances, hier_As, n_p, last_act_fn, batch_norm):
    error = np.zeros(N_EXPS)

    for i,x in enumerate(signals):
        x_n = utils.RandomGraphSignal.add_noise(x, n_p)
        for j, exp in enumerate(EXPERIMENTS):
            cont = i*len(EXPERIMENTS)+j
            dec = GraphDeepDecoder(descendances[j], hier_As[j], sizes[j],
                            n_channels=exp['arch'], upsampling=exp['ups'],
                            last_act_fun=last_act_fn, batch_norm=batch_norm)
            dec.build_network()
            x_est, _ = dec.fit(x_n)
            error[cont] = np.sum(np.square(x-x_est))/np.square(np.linalg.norm(x))
            print('Signal: {} Scenario: {} Error: {:.4f}'
                                .format(id, cont+1, error[cont]))
    return error

def print_results(mean_mse, median_mse, n_p):
    for i, exp in enumerate(EXPERIMENTS):
        print('{}. (EXP {}) '.format(i, exp))
        print('\tMean MSE: {}\tMedian MSE: {}'
                            .format(mean_mse[i], median_mse[i]))

def save_results(data):
    if not os.path.isdir('./results/test_input'):
        os.makedirs('./results/test_input')
    timestamp = datetime.datetime.now().strftime("%Y_%m_%d-%H_%M")
    path = './results/test_input/input_{}'.format(timestamp)
    np.save(path, data)
    print('SAVED as:', path)

if __name__ == '__main__':
    data = {}
    data['L'] = L = 6
    data['last_act_fn'] = nn.Sigmoid()
    data['batch_norm'] = True
    data['EXPERIMENTS'] = EXPERIMENTS
    data['INPUTS'] = INPUTS
    
    # Graph parameters
    G_params = {}
    G_params['type'] = 'SBM' # SBM or ER
    G_params['N'] = N = 256
    G_params['k'] = k = 4
    G_params['p'] = 0.15
    G_params['q'] = 0.01/k

    # Set seeds
    utils.RandomGraphSignal.set_seed(SEED)
    GraphDeepDecoder.set_seed(SEED)

    start_time = time.time()
    G = utils.create_graph(G_params, seed=SEED)
    sizes, descendances, hier_As = compute_clusters(G, G_params['k'])
    data['g_params'] = G_params

    error = np.zeros((N_SIGNALS, N_EXPS))
    with Pool() as pool:
        for j in range(N_SIGNALS):
            signals = [create_signal(s_in,G,L,k).x for s_in in INPUTS]
            result = pool.apply_async(test_input,
                        args=[j, signals, sizes, descendances, hier_As, n_p,
                                data['last_act_fn'], data['batch_norm']])
        for j in range(N_SIGNALS):
            error[j,:] = result.get()

    print_results(np.mean(error, axis=0), np.median(error, axis=0), n_p)
    data['mse'] = error
    save_results(data)

    print('--- {} minutes ---'.format((time.time()-start_time)/60))
    plt.show()
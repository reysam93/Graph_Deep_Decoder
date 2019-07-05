import sys
import os
import time, datetime
from multiprocessing import Pool, cpu_count
sys.path.insert(0, 'graph_deep_decoder')
from graph_deep_decoder import utils
from graph_deep_decoder.architecture import GraphDeepDecoder
from scipy.sparse.csgraph import dijkstra
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn

# Constants
N_SIGNALS = 200
SEED = 15
p_miss = 0.1

EXPERIMENTS = [{'ups': 'original', 'arch': [3,3,3], 't': [4,16,64,256]},
               {'ups': 'no_A', 'arch': [3,3,3], 't': [4,16,64,256]},
               {'ups': 'binary', 'arch': [3,3,3], 't': [4,16,64,256]},
               {'ups': 'weighted', 'arch': [3,3,3], 't': [4,16,64,256]},
               {'ups': 'original', 'arch': [6,6,6], 't': [4,16,64,256]},
               {'ups': 'no_A', 'arch': [6,6,6], 't': [4,16,64,256]},
               {'ups': 'binary', 'arch': [6,6,6], 't': [4,16,64,256]},
               {'ups': 'weighted', 'arch': [6,6,6], 't': [4,16,64,256]},
               {'ups': 'original', 'arch': [4,4,4,4], 't': [4,16,32,64,256]},
               {'ups': 'no_A', 'arch': [4,4,4,4], 't': [4,16,32,64,256]},
               {'ups': 'binary', 'arch': [4,4,4,4], 't': [4,16,32,64,256]},
               {'ups': 'weighted', 'arch': [4,4,4,4], 't': [4,16,32,64,256]}]
N_EXPS = len(EXPERIMENTS)

def compute_clusters(G, root_clust):
    sizes = []
    descendances = []
    hier_As = []
    for exp in EXPERIMENTS:
        cluster = utils.MultiRessGraphClustering(G, exp['t'], root_clust)
        sizes.append(cluster.clusters_size)
        descendances.append(cluster.compute_hierarchy_descendance())
        hier_As.append(cluster.compute_hierarchy_A(exp['ups']))
    return sizes, descendances, hier_As

def create_signal(signal_type, G, L, k, D=None):
    if signal_type == 'linear':
        signal = utils.DifussedSparseGS(G,L,k)
    elif signal_type == 'non-linear':
        signal = utils.NonLinealDSGS(G,L,k,D)
    elif signal_type == 'median':
        signal = utils.MedianDSGS(G,L,k)
    elif signal_type == 'comb':
        signal = utils.NLCombinationsDSGS(G,L,k)
    else:
        raise RuntimeError('Unknown signal type')
    signal.signal_to_0_1_interval()
    signal.to_unit_norm()
    return signal

def test_input(id, x, sizes, descendances, hier_As, p_miss, 
                last_act_fn, batch_norm):
    error = np.zeros(N_EXPS)

    inp_mask = utils.GraphSignal.generate_inpaint_mask(x, p_miss)
    clean_error = np.sum(np.square(x-x*inp_mask))/np.square(np.linalg.norm(x))
    for i, exp in enumerate(EXPERIMENTS):
        dec = GraphDeepDecoder(descendances[i], hier_As[i], sizes[i],
                        n_channels=exp['arch'], upsampling=exp['ups'],
                        last_act_fun=last_act_fn, batch_norm=batch_norm)
        dec.build_network()
        x_est, _ = dec.fit(x, mask=inp_mask, n_iter=4000)
        error[i] = np.sum(np.square(x-x_est))/np.square(np.linalg.norm(x))
        print('Signal: {} Scenario: {} Error: {:.4f}'
                            .format(id, i+1, error[i]))
    return error, clean_error

def print_results(error, mean_clean_err, p_miss):
    mean_mse = np.mean(error, axis=0)
    median_mse = np.median(error, axis=0)
    std = np.std(error, axis=0)
    print('Clean Err: {} (p_miss: {})'.format(mean_clean_err, p_miss))
    for i, exp in enumerate(EXPERIMENTS):
        print('{}. EXP: {}'.format(i+1, exp))
        print('\tMean MSE: {}\tMedian MSE: {}\tSTD: {}'
                        .format(mean_mse[i], median_mse[i], std[i]))

def save_results(data):
    if not os.path.isdir('./results/test_inpaint'):
        os.makedirs('./results/test_inpaint')
    timestamp = datetime.datetime.now().strftime("%Y_%m_%d-%H_%M")
    path = './results/test_inpaint/inpaint_miss_{}_{}'.format(p_miss, timestamp)
    np.save(path, data)
    print('SAVED as:', path)

if __name__ == '__main__':
    data = {}
    data['L'] = L = 6
    data['last_act_fn'] = nn.Sigmoid()
    data['batch_norm'] = True
    data['EXPERIMENTS'] = EXPERIMENTS
    data['p_miss'] = p_miss
    data['input'] = 'median'

    # Graph parameters
    G_params = {}
    G_params['type'] = 'SBM' # SBM or ER
    G_params['N'] = N = 256
    G_params['k'] = k = 4
    G_params['p'] = 0.15
    G_params['q'] = 0.01/k

    # Set seeds
    utils.GraphSignal.set_seed(SEED)
    GraphDeepDecoder.set_seed(SEED)

    start_time = time.time()
    G = utils.create_graph(G_params, seed=SEED)
    sizes, descendances, hier_As = compute_clusters(G, G_params['k'])
    data['g_params'] = G_params

    error = np.zeros((N_SIGNALS, N_EXPS))
    clean_err = np.zeros(N_SIGNALS)
    results = []
    with Pool(processes=cpu_count()) as pool:
        for j in range(N_SIGNALS):
            x = create_signal(data['input'],G,L,k).x
            results.append(pool.apply_async(test_input,
                        args=[j, x, sizes, descendances, hier_As, p_miss,
                                data['last_act_fn'], data['batch_norm']]))
        for j in range(N_SIGNALS):
            error[j,:], clean_err[j] = results[j].get()

    print_results(error, np.median(clean_err), p_miss)
    data['mse'] = error
    data['clean_err'] = clean_err
    save_results(data)

    print('--- {} minutes ---'.format((time.time()-start_time)/60))
    plt.show()
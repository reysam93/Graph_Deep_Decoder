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
N_SIGNALS = 100 
SEED = 15
N_P = [0, .05, .1, .2, .3, .5]
EXPERIMENTS = [{'ups': 'original', 'arch': [3,3,3], 't': [4,16,64,None], 'gamma': None},
               {'ups': 'no_A', 'arch': [3,3,3], 't': [4,16,64,None], 'gamma': None},
               {'ups': 'weighted', 'arch': [3,3,3], 't': [4,16,64,None], 'gamma': 0},
               {'ups': 'weighted', 'arch': [3,3,3], 't': [4,16,64,None], 'gamma': .75},
               {'ups': 'original', 'arch': [6,6,6], 't': [4,16,64,None], 'gamma': None},
               {'ups': 'no_A', 'arch': [6,6,6], 't': [4,16,64,None], 'gamma': None},
               {'ups': 'weighted', 'arch': [6,6,6], 't': [4,16,64,None], 'gamma': 0},
               {'ups': 'weighted', 'arch': [6,6,6], 't': [4,16,64,None], 'gamma': .75},
               {'ups': 'original', 'arch': [4,4,4,4], 't': [4,16,64,128,None], 'gamma': None},
               {'ups': 'no_A', 'arch': [4,4,4,4], 't': [4,16,64,128,None], 'gamma': None},
               {'ups': 'weighted', 'arch': [4,4,4,4], 't': [4,16,64,128,None], 'gamma': 0},
               {'ups': 'weighted', 'arch': [4,4,4,4], 't': [4,16,64,128,None], 'gamma': .75},]

FMTS = ['o-', '^-', '+-', 'x-', 'o--', '^--', '+--', 'x--', 'o:', '^:', '+:', 'x:']

G_PARAMS = [{'type': 'SBM','N': 256,'k': 4,'p': 0.15,'q': 0.01/4},
            {'type': 'SBM','N': 512,'k': 4,'p': 0.075,'q': 0.001},
            {'type': 'SBM','N': 1024,'k': 10,'p': 0.1,'q': 0.0005}]

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

def denoise(x, sizes, descendances, hier_As, n_p, last_act_fn, batch_norm):
    error = np.zeros(len(EXPERIMENTS))
    params = np.zeros(len(EXPERIMENTS))
    x_n = utils.RandomGraphSignal.add_noise(x, n_p)
    for i, exp in enumerate(EXPERIMENTS):
        dec = GraphDeepDecoder(descendances[i], hier_As[i], sizes[i],
                        n_channels=exp['arch'], upsampling=exp['ups'],
                        gamma=exp['gamma'], last_act_fun=last_act_fn,
                        batch_norm=batch_norm)
        dec.build_network()
        x_est, _ = dec.fit(x_n)
        error[i] = np.sum(np.square(x-x_est))/np.linalg.norm(x)
        params[i] = dec.count_params()
        print('\tScenario {}: Error: {:.4f}'
                            .format(i, error[i]))
    return error, params

def print_results(N, mse_est, params, n_p):
    mean_mse = np.mean(mse_est, axis=0)
    median = np.median(mse_est, axis=0)
    std = np.std(mse_est, axis=0)
    print('NOISE POWER:', n_p)
    for i, exp in enumerate(EXPERIMENTS):
        print('{}. (EXP {}) '.format(i, exp))
        print('\tMean MSE: {}\t Median: {}\tSTD: {}\tParams: {}\tCompression: {}'
                            .format(mean_mse[i], median[i], std[i], params[i], N/params[i]))

def plot_results(mean_mse):
    plt.figure()
    for i in range(len(EXPERIMENTS)):
        plt.plot(N_P, mean_mse[:,i], FMTS[i])
    
    plt.xlabel('Noise Power')
    plt.ylabel('Mean MSE')
    legend = [str(exp) for exp in EXPERIMENTS]
    plt.legend(legend)

def save_results(data):
    N = data['g_params']['N']
    if not os.path.isdir('./results/denoising'):
        os.makedirs('./results/denoising')
    timestamp = datetime.datetime.now().strftime("%Y_%m_%d-%H_%M")
    path = './results/denoising/denoise_N_{}_{}'.format(N, timestamp)
    np.save(path, data)
    print('SAVED as:', path)

if __name__ == '__main__':
    data = {}
    data['L'] = 6
    data['last_act_fn'] = nn.Sigmoid()
    data['batch_norm'] = True
    data['FMTS'] = FMTS
    data['EXPERIMENTS'] = EXPERIMENTS
    data['N_P'] = N_P

    # Set seeds
    utils.GraphSignal.set_seed(SEED)
    GraphDeepDecoder.set_seed(SEED)

    start_time = time.time()
    for g_params in G_PARAMS:
        G = utils.create_graph(g_params, seed=SEED)
        sizes, descendances, hier_As = compute_clusters(G, g_params['k'])
        data['g_params'] = g_params
        print('Graph', g_params)

        results = []
        mse_est = np.zeros((len(N_P), N_SIGNALS, len(EXPERIMENTS)))
        for i, n_p in enumerate(N_P):    
            with Pool() as pool:
                for j in range(N_SIGNALS):
                    signal = utils.DifussedSparseGS(G,data['L'],g_params['k'])
                    signal.signal_to_0_1_interval()
                    signal.to_unit_norm()
                    results.append(pool.apply_async(denoise,
                                args=[signal.x, sizes, descendances, hier_As, n_p,
                                        data['last_act_fn'], data['batch_norm']]))
                for j in range(N_SIGNALS):
                    mse_est[i,j,:], n_params = results[j].get()
                    print('Signal',j)

            print_results(g_params['N'], mse_est[i,:,:], n_params, n_p)
        plot_results(np.mean(mse_est, axis=1))
        data['mse'] = mse_est
        save_results(data)

    print('--- {} minutes ---'.format((time.time()-start_time)/60))
    plt.show()
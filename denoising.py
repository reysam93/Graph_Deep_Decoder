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
N_P = [0, .001, .005, .01, .05, .1, .5]
EXPERIMENTS = [{'ups': 'original', 'arch': [2,2,2], 't': [4,16,64,None], 'gamma': None},
               {'ups': 'no_A', 'arch': [2,2,2], 't': [4,16,64,None], 'gamma': None},
               {'ups': 'weighted', 'arch': [2,2,2], 't': [4,16,64,None], 'gamma': 0},
               {'ups': 'weighted', 'arch': [2,2,2], 't': [4,16,64,None], 'gamma': .75},
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

def denoise(x, sizes, descendances, hier_As, n_p):
    mse_est = np.zeros(len(EXPERIMENTS))
    params = np.zeros(len(EXPERIMENTS))
    x_n = utils.DifussedSparseGraphSignal.add_noise(x, n_p)
    for i, exp in enumerate(EXPERIMENTS):
        dec = GraphDeepDecoder(descendances[i], hier_As[i], sizes[i],
                        n_channels=exp['arch'], upsampling=exp['ups'],
                        gamma=exp['gamma'])
        dec.build_network()
        x_est, _ = dec.fit(x_n)
        mse_est[i] = np.mean(np.square(x-x_est))/np.linalg.norm(x)
        params[i] = dec.count_params()
    return mse_est, params

def print_results(N, mean_mse, params, n_p):
    print('NOISE POWER:', n_p)
    for i, exp in enumerate(EXPERIMENTS):
        print('{}. (EXP {}) '.format(i, exp))
        print('\tMean MSE: {}\tParams: {}\tCompression: {}'
                            .format(mean_mse[i], params[i], N/params[i]))

def plot_results(mean_mse):
    plt.figure()
    for i in range(len(EXPERIMENTS)):
        plt.plot(N_P, mean_mse[:,i], FMTS[i])
    
    plt.xlabel('Noise Power')
    plt.ylabel('Mean MSE')
    legend = [str(exp) for exp in EXPERIMENTS]
    plt.legend(legend)

def save_results(mse_est, N):
    if not os.path.isdir('./results/denoising'):
        os.makedirs('./results/denoising')
    timestamp = datetime.datetime.now().strftime("%Y_%m_%d-%H_%M")
    np.save('./results/denoising/denoise_N_{}_{}'.format(N, timestamp), mse_est)


if __name__ == '__main__':
    L = 5

    # Set seeds
    utils.DifussedSparseGraphSignal.set_seed(SEED)
    GraphDeepDecoder.set_seed(SEED)

    start_time = time.time()
    for g_params in G_PARAMS:
        G = utils.create_graph(g_params, seed=SEED)
        sizes, descendances, hier_As = compute_clusters(G, g_params['k'])
        print('Graph', g_params)

        mse_est = np.zeros((len(N_P), N_SIGNALS, len(EXPERIMENTS)))
        for i, n_p in enumerate(N_P):    
            with Pool() as pool:
                for j in range(N_SIGNALS):
                    signal = utils.DifussedSparseGraphSignal(G,L,g_params['k'])
                    signal.to_unit_norm()
                    result = pool.apply_async(denoise,
                                args=[signal.x, sizes, descendances, hier_As, n_p])
                for j in range(N_SIGNALS):
                    mse_est[i,j,:], n_params = result.get()

            print_results(g_params['N'], np.mean(mse_est[i,:,:], axis=0), n_params, n_p)
        plot_results(np.mean(mse_est, axis=1))
        save_results(mse_est, g_params['N'])

    print('--- {} minutes ---'.format((time.time()-start_time)/60))
    plt.show()
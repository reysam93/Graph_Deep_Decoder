import sys
import os

import numpy as np
import torch.nn as nn
from torch import manual_seed

sys.path.insert(0, '../graph_deep_decoder')
from graph_deep_decoder import datasets as ds
from graph_deep_decoder import utils


REAL_DATA = True
DATASET_PATH = 'dataset/graphs.mat'
# DATASET_PATH = 'dataset/all_data_eigA_x8_S3_minN25.mat'
MAX_SIGNALS = 100
MIN_SIZE = 50
MAX_SIZE = 150
ATTR = 8

MAX_SM = None


if __name__ == '__main__':
    # Set seeds
    # seed = None
    seed = 15
    np.random.seed(seed)
    manual_seed(seed)

    # Graph prams
    Gs = {'type': ds.SBM,
          'N': 256,
          'k': 4,
          'p': 0.25,
          'q': [[0, 0.0075, 0, 0.0],
                [0.0075, 0, 0.004, 0.0025],
                [0, 0.004, 0, 0.005],
                [0, 0.0025, 0.005, 0]],
          'm': 1,
          'm0': 1,
          'type_z': ds.RAND}


    # Signal params
    Signals = {}
    Signals['n_signals'] = 50
    Signals['type'] = ds.SigType.DS
    Signals['non_lin'] = ds.NonLin.MEDIAN
    Signals['deltas'] = Gs['k']
    Signals['L'] = 6
    Signals['noise'] = 0
    Signals['to_0_1'] = False
    Signals['pos_coefs'] = False

    if REAL_DATA:
        Gs, signals = utils.read_graphs(DATASET_PATH, ATTR, MIN_SIZE,
                                        MAX_SIGNALS, max_size=MAX_SIZE,
                                        max_smooth=MAX_SM, center=False)
        Signals['n_signals'] = len(Gs)

    sm = np.zeros(Signals['n_signals'])
    tv = np.zeros(Signals['n_signals'])
    tv_l2 = np.zeros(Signals['n_signals'])
    bl = np.zeros(Signals['n_signals'])
    size = np.zeros(Signals['n_signals'])
    n_zeros = np.zeros(Signals['n_signals'])
    for i in range(Signals['n_signals']):
        if not REAL_DATA:
            G = ds.create_graph(Gs, seed)
            signal = ds.GraphSignal.create(Signals['type'], G,
                                           Signals['non_lin'], Signals['L'],
                                           Signals['deltas'],
                                           pos_coefs=Signals['pos_coefs'],
                                           to_0_1=Signals['to_0_1'])
        else:
            G, signal = Gs[i], signals[i]
            # signal.plot(True)

        # if i is 0:
        #     signal.plot(True)

        sm[i] = signal.smoothness()
        tv[i] = signal.total_variation()
        tv_l2[i] = signal.total_variation(norm=2)
        bl[i] = signal.check_bandlimited(coefs=0.25)
        n_zeros[i] = np.sum(signal.x == 0)
        size[i] = G.N
        print('Signal {}: N: {} SM: {} TV: {} TVl2: {} BL: {}'
              .format(i, G.N, sm[i], tv[i], tv_l2[i], bl[i]))

    print('Mean: SM: {} - TV: {} - TVl2: {} - BL: {} - N: {} - Zeros: {}'
          .format(np.mean(sm), np.mean(tv), np.mean(tv_l2), np.mean(bl),
                  np.mean(size), np.mean(n_zeros)))
    # signal.plot(show=False)
    # signal.plot_freq_resp()

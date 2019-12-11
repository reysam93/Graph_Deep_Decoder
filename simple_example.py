import sys

import numpy as np
from torch import manual_seed
import torch.nn as nn

from graph_deep_decoder import datasets as ds
from graph_deep_decoder import graph_clustering as gc
from graph_deep_decoder.architecture import GraphDeepDecoder
from graph_deep_decoder.model import Model
from graph_deep_decoder import utils


if __name__ == '__main__':
    # Set seeds
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
          'type_z': ds.RAND}

    # Signal params
    Sign = {'L': 6,
            'deltas': Gs['k'],
            'noise': 0.3,
            'to_0_1': True}

    Net = {}
    Net['laf'] = nn.Tanh()
    Net['af'] = nn.Tanh()  # nn.CELU()
    Net['lr'] = 0.001
    Net['dr'] = 1

    # Arch params
#     nodes = [4, 8, 32, 256, 256, 256]
#     fts = [3, 3, 3, 3, 3, 1]
#     nodes = [256]*6
    nodes = [4, 8, 16, 32, 64] + [256]*3
    fts = [15]*7 + [1]

    G = ds.create_graph(Gs, seed)
    cluster = gc.MultiResGraphClustering(G, nodes, Gs['k'], up_method=gc.WEI)
#     cluster.plot_labels()

    dec = GraphDeepDecoder(fts, cluster.sizes, cluster.Us, As=cluster.As,
                           ups=gc.WEI, act_fn=Net['af'],
                           last_act_fn=Net['laf'])
    signal = ds.GraphSignal.create(ds.LINEAR, G, Sign['L'],
                                   Sign['deltas'], to_0_1=Sign['to_0_1'])
    signal.to_unit_norm()
#     signal.plot()

    x_n = ds.GraphSignal.add_noise(signal.x, Sign['noise'])
    model = Model(dec, learning_rate=Net['lr'], decay_rate=Net['dr'],
                  verbose=True, eval_freq=100, epochs=5000)
    train_err, val_err, best_epoch = model.fit(x_n, x=signal.x)
    print('Best epoch:', best_epoch)
    utils.plot_overfitting(train_err, val_err)
    node_err, err = model.test(signal.x)
    print('Node Err:', node_err, 'Err:', err, 'N Params', model.count_params())

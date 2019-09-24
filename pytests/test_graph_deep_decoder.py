import unittest
import sys
import numpy as np
from torch.nn import Sequential
from torch import Tensor

sys.path.insert(0, '.')
sys.path.insert(0, '.\graph_deep_decoder')
from graph_deep_decoder import utils
from graph_deep_decoder.architecture import GraphUpsampling

SEED = 15

class GraphUpsamplingTest(unittest.TestCase):
    def setUp(self):
        np.random.seed(SEED)
        G_params = {}
        G_params['type'] = 'SBM'
        G_params['N']  = self.N = 256
        G_params['k']  = self.k = 4
        G_params['p'] = 0.15
        G_params['q'] = 0.01/4
        type_z = 'contiguos'
        G = utils.create_graph(G_params, SEED, type_z=type_z)
        nodes = [4, 16, 64, 256]
        ups = 'weighted'
        self.cluster = utils.MultiRessGraphClustering(G, nodes, k=4)
        self.cluster.compute_hierarchy_descendance()
        self.cluster.compute_hierarchy_A(ups)
        self.model = Sequential()
        for i in range(len(nodes)-1):
            self.add_layer(GraphUpsampling(self.cluster.descendance[i], nodes[i], 
            self.cluster.hier_A[i+1], ups, gamma=1))

    def add_layer(self, module):
        self.model.add_module(str(len(self.model) + 1), module)

    def test_1D_upsampling(self):
        expected_result = self.cluster.labels[0].astype(np.float32)
        expected_result = Tensor(expected_result).view([1, 1, self.N])
        input = np.unique(self.cluster.labels[0]).astype(np.float32)
        input = Tensor(input).view([1, 1, self.k])
        result = self.model(input)
        self.assertTrue(result.equal(expected_result))

    def test_upsampling_with_channels(self):
        n_chans = 3
        expected_result = self.cluster.labels[0].astype(np.float32)
        expected_result = np.repeat([expected_result], n_chans, axis=0)
        expected_result = Tensor(expected_result).view([1, n_chans, self.N])
        input = np.unique(self.cluster.labels[0]).astype(np.float32)
        input = np.repeat([input], n_chans, axis=0)
        input = Tensor(input).view([1, n_chans, self.k])
        result = self.model(input)

        print(input)
        print(result)
        
        self.assertTrue(result.equal(expected_result))

        

if __name__ == "__main__":
    unittest.main()
import unittest
import sys
import numpy as np
from torch.nn import Sequential
from torch import Tensor, zeros, matrix_power, randn

sys.path.insert(0, '.')
sys.path.insert(0, './graph_deep_decoder')
from graph_deep_decoder import datasets as ds
from graph_deep_decoder import graph_clustering as gc
from graph_deep_decoder.architecture import MeanUps, GFUps, NVGFUps


SEED = 15


class GraphClustSizesTest(unittest.TestCase):
    def setUp(self):
        self.G_params = {}
        self.G_params['type'] = ds.SBM
        self.G_params['N'] = 256
        self.G_params['k'] = 4
        self.G_params['p'] = 0.15
        self.G_params['q'] = 0.01/4
        self.G_params['type_z'] = ds.CONT
        self.G = ds.create_graph(self.G_params, seed=SEED)

    def test_wrong_sizes(self):
        nodes = [4, 4, 16, 32, 32, 64, 64, 256, 64, 256, 256]
        try:
            gc.MultiResGraphClustering(self.G, nodes, k=4)
            self.fail()
        except:
            pass

    def test_all_sizes_diff(self):
        nodes = [4, 16, 32, 64, 256]
        cluster = gc.MultiResGraphClustering(self.G, nodes, k=4)
        self.assertEqual(len(nodes), len(cluster.sizes))
        self.assertEqual(len(nodes)-1, len(cluster.Us))

    def test_repeated_sizes(self):
        nodes = [4, 4, 16, 32, 32, 64, 256, 256, 256]
        cluster = gc.MultiResGraphClustering(self.G, nodes, k=4)
        self.assertEqual(len(nodes), len(cluster.sizes))
        self.assertEqual(len(nodes)-1, len(cluster.Us))

    def test_all_sizes_repeated(self):
        nodes = [256, 256, 256]
        cluster = gc.MultiResGraphClustering(self.G, nodes, k=4)
        self.assertEqual(len(nodes), len(cluster.sizes))
        for U in cluster.Us:
            self.assertEqual(U, None)


class UpsamplingMatricesTest(unittest.TestCase):
    def setUp(self):
        np.random.seed(SEED)
        self.G_params = {}
        self.G_params['type'] = ds.SBM
        self.G_params['N'] = 256
        self.G_params['k'] = 4
        self.G_params['p'] = 0.15
        self.G_params['q'] = 0.01/4
        self.nodes_enc = [4, 16, 32, 64, 256]

    def test_cont_nodes(self):
        self.G_params['type_z'] = ds.CONT
        G = ds.create_graph(self.G_params, seed=SEED)
        cluster = gc.MultiResGraphClustering(G, self.nodes_enc, k=4)
        out = np.unique(cluster.labels[0])
        for U in cluster.Us:
            out = U.dot(out)
        self.assertTrue(np.array_equal(out, cluster.labels[0]))

    def test_alt_nodes(self):
        self.G_params['type_z'] = ds.ALT
        G = ds.create_graph(self.G_params, seed=SEED)
        cluster = gc.MultiResGraphClustering(G, self.nodes_enc, k=4)
        out = np.unique(cluster.labels[0])
        for U in cluster.Us:
            out = U.dot(out)
        self.assertTrue(np.array_equal(out, cluster.labels[0]))

    def test_cont_nodes(self):
        self.G_params['type_z'] = ds.RAND
        G = ds.create_graph(self.G_params, seed=SEED)
        cluster = gc.MultiResGraphClustering(G, self.nodes_enc, k=4)
        out = np.unique(cluster.labels[0])
        for U in cluster.Us:
            out = U.dot(out)
        self.assertTrue(np.array_equal(out, cluster.labels[0]))


class MeanUpsModulesTest(unittest.TestCase):
    def setUp(self):
        np.random.seed(SEED)
        G_params = {}
        G_params['type'] = ds.SBM
        G_params['N'] = 256
        G_params['k'] = 4
        G_params['p'] = 0.15
        G_params['q'] = 0.01/4
        G_params['type_z'] = ds.RAND
        G = ds.create_graph(G_params, seed=SEED)
        nodes_dec = [4, 16, 32, 64, 256]
        self.N = nodes_dec[-1]
        self.k = nodes_dec[0]
        self.cluster = gc.MultiResGraphClustering(G, nodes_dec, k=4)
        self.model = Sequential()
        for i, U in enumerate(self.cluster.Us):
            self.add_layer(MeanUps(U, self.cluster.As[i], gamma=1))

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
        n_chans = 5
        expected_result = self.cluster.labels[0].astype(np.float32)
        expected_result = np.repeat([expected_result], n_chans, axis=0)
        expected_result = Tensor(expected_result).view([1, n_chans, self.N])
        input = np.unique(self.cluster.labels[0]).astype(np.float32)
        input = np.repeat([input], n_chans, axis=0)
        input = Tensor(input).view([1, n_chans, self.k])
        result = self.model(input)
        self.assertTrue(result.equal(expected_result))

    def test_upsampling_with_channels_and_samples(self):
        n_chans = 3
        n_samples = 5
        input_aux = np.unique(self.cluster.labels[0]).astype(np.float32)
        input_aux = np.repeat([input_aux], n_chans, axis=0)
        input = zeros([n_samples, n_chans, self.k])
        try:
            result = self.model(input)
            self.fail()
        except:
            pass


class GFUpsTest(unittest.TestCase):
    def setUp(self):
        G_params = {}
        G_params['type'] = ds.SBM
        G_params['N'] = 256
        G_params['k'] = 4
        G_params['p'] = 0.15
        G_params['q'] = 0.01/4
        G_params['type_z'] = ds.RAND
        G = ds.create_graph(G_params, seed=SEED)
        nodes = [4, 16, 32, 256]
        self.K = 3
        cluster = gc.MultiResGraphClustering(G, nodes, k=4)
        self.ups = []
        self.As = []
        self.Us = []
        for i in range(len(cluster.Us)):
            self.ups.append(GFUps(cluster.Us[i], cluster.As[i],
                            self.K))
            self.As.append(Tensor(cluster.As[i]))
            self.Us.append(Tensor(cluster.Us[i]))

    def create_H(self, hs, A):
        H = zeros(A.size())
        for i in range(self.K):
            H += hs[i]*matrix_power(A, i)
        return H

    def test_upsampling_with_channels(self):
        n_chans = 5
        for i, up in enumerate(self.ups):
            inp = randn(1, n_chans, self.Us[i].size()[1])
            hs = Tensor(up.hs.data)
            out = up(inp)
            H = self.create_H(hs, self.As[i])
            U_T = self.Us[i].t()
            exp_out = inp[0, :, :].mm(U_T.mm(H.t()))
            exp_out = exp_out.view([1, n_chans, self.As[i].size()[0]])
            self.assertTrue(out.eq(exp_out).all())


class NVGFUpsTest(unittest.TestCase):
    def setUp(self):
        np.random.seed(SEED)
        G_params = {}
        G_params['type'] = ds.SBM
        G_params['N'] = 256
        G_params['k'] = 4
        G_params['p'] = 0.15
        G_params['q'] = 0.01/4
        G_params['type_z'] = ds.RAND
        G = ds.create_graph(G_params, seed=SEED)
        nodes = [4, 16, 32, 256]
        self.K = 3
        cluster = gc.MultiResGraphClustering(G, nodes, k=4)
        self.ups = []
        self.As = []
        self.Us = []
        for i in range(len(cluster.Us)):
            self.ups.append(NVGFUps(cluster.Us[i], cluster.As[i],
                            self.K))
            self.As.append(Tensor(cluster.As[i]))
            self.Us.append(Tensor(cluster.Us[i]))

    def create_H(self, hs, A):
        H = zeros(A.size())
        for k in range(self.K):
            Apow = matrix_power(A, k)
            for j in range(H.size()[0]):
                H[j, :] += hs[k, j]*Apow[j, :]
        return H

    def test_upsampling_with_channels(self):
        n_chans = 5
        for i, up in enumerate(self.ups):
            inp = randn(1, n_chans, self.Us[i].size()[1])
            up.hs.data.uniform_()
            hs = Tensor(up.hs.data)
            out = up(inp)
            H = self.create_H(hs, self.As[i])
            U_T = self.Us[i].t()
            exp_out = inp[0, :, :].mm(U_T.mm(H.t()))
            exp_out = exp_out.view([1, n_chans, self.As[i].size()[0]])
            self.assertTrue(out.eq(exp_out).all())


if __name__ == "__main__":
    unittest.main()

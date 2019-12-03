import unittest
import sys
import numpy as np

sys.path.insert(0, '.')
# sys.path.insert(0, './graph_deep_decoder')
from graph_deep_decoder import datasets as ds


SEED = 15


class SBMGraphCreationTest(unittest.TestCase):
    def setUp(self):
        np.random.seed(SEED)
        self.G_params = {}
        self.G_params['type'] = ds.SBM
        self.G_params['N'] = self.N = 256
        self.G_params['k'] = self.k = 4
        self.G_params['p'] = 0.20
        self.G_params['q'] = 0.015/4

    def test_cont_nodes(self):
        self.G_params['type_z'] = ds.CONT
        pass

    def test_alt_nodes(self):
        pass

    def test_rand_nodes(self):
        pass


class GraphSignalTest(unittest.TestCase):
    def setUp(self):
        np.random.seed(SEED)
        self.G_params = {}
        self.G_params['type'] = ds.SBM
        self.G_params['N'] = 256
        self.G_params['k'] = 4
        self.G_params['p'] = 0.3
        self.G_params['q'] = 0.05
        self.G_params['type_z'] = ds.RAND
        self.Gx = ds.create_graph(self.G_params, seed=SEED)

    def test_to_unit_norm(self):
        signals = 5
        L = 6
        n_delts = 4
        for i in range(signals):
            data = ds.DifussedSparseGS(self.Gx, L, n_delts)
            data.to_unit_norm()
            self.assertAlmostEqual(np.linalg.norm(data.x), 1)

    def test_add_noise(self):
        signals = 5
        lim = 0.1
        noise = np.random.rand(signals)/2+0.25
        L = 6
        n_delts = 4
        for i in range(signals):
            data = ds.DifussedSparseGS(self.Gx, L, n_delts)
            data.to_unit_norm()
            x_n = ds.GraphSignal.add_noise(data.x, noise[i])
            noise_est = np.linalg.norm(data.x-x_n)**2
            print(noise[i], noise_est)
            self.assertGreaterEqual(noise_est, noise[i]-lim)
            self.assertLessEqual(noise_est, noise[i]+lim)


if __name__ == "__main__":
    unittest.main()

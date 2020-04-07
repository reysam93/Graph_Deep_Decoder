import unittest
import sys
import numpy as np

sys.path.insert(0, '.')
from graph_deep_decoder.architecture import GraphDecoder
import torch

SEED = 15


class GraphDecoderTest(unittest.TestCase):
    def setUp(self):
        np.random.seed(SEED)

    def test_expected_output(self):
        N = 5
        k = 10
        H = np.random.rand(N, N)
        H = torch.Tensor((H+H.T)/2)
        dec = GraphDecoder(k, H)
        out1 = dec(dec.input).squeeze()
        weights = dec.conv.weight.data.squeeze().t()
        out2 = dec.relu(H.mm(weights)).mv(dec.v).squeeze()
        self.assertTrue(out1.allclose(out2))

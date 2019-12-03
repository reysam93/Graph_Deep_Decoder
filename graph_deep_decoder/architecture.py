import copy
import torch
from torch import manual_seed, nn, Tensor, optim, no_grad
import torch.nn as nn
import numpy as np

from graph_deep_decoder.graph_clustering import NONE, REG, NO_A, BIN, WEI


# Error messages
ERR_DIFF_N_LAYERS = 'Length of the nodes and features vector must be the same'
ERR_A_NON_SYM = 'Matrix A for upsampling should be symmetric'
ERR_WRONG_METHOD = 'Wrong combination of methods for upsampling'
ERR_WRONG_INPUT_SIZE = 'Number of input samples must be 1'
ERR_UNK_UPS = 'Unkown upsampling method {}'


class GraphDeepDecoder(nn.Module):
    def __init__(self,
                 # Decoder args
                 features, nodes, Us,
                 # Optional args
                 As=None,  ups=WEI,
                 gamma=0.5, batch_norm=True,
                 # Activation functions
                 act_fn=nn.Tanh(), last_act_fn=nn.Tanh()):
        assert len(features) == len(nodes), ERR_DIFF_N_LAYERS
        assert ups in [None, REG, NO_A, BIN, WEI], ERR_UNK_UPS.format(ups)

        super(GraphDeepDecoder, self).__init__()
        self.model = nn.Sequential()
        self.fts = features
        self.nodes = nodes
        self.Us = Us
        self.As = As
        self.ups = ups
        self.kernel_size = 1
        self.gamma = gamma
        self.batch_norm = batch_norm
        self.act_fn = act_fn
        self.last_act_fn = last_act_fn
        self.build_network()

        if self.ups is not None:
            shape = [1, self.fts[0], self.nodes[0]]
        else:
            shape = [1, self.fts[0], self.nodes[-1]]
        self.input = Tensor(torch.zeros(shape)).data.normal_()

    def add_layer(self, module):
        self.model.add_module(str(len(self.model) + 1), module)

    def build_network(self):
        ups_skip = 0
        for l in range(len(self.fts)-1):
            self.add_layer(nn.Conv1d(self.fts[l], self.fts[l+1],
                           self.kernel_size, bias=False))

            if self.nodes[l] < self.nodes[l+1]:
                if self.As:
                    A = self.As[l+1-ups_skip]
                else:
                    A = None
                self.add_layer(GraphUpsampling(self.Us[l-ups_skip],
                                               A, self.gamma, self.ups))
            else:
                ups_skip += 1

            if l < (len(self.fts)-2):
                # Not the last layer
                if self.act_fn is not None:
                    self.add_layer(self.act_fn)
                if self.batch_norm:
                    self.add_layer(nn.BatchNorm1d(self.fts[l+1]))
            else:
                # Last layer
                if self.last_act_fn is not None:
                    self.add_layer(self.last_act_fn)
        return self.model

    def forward(self, x):
            return self.model(x)

    def count_params(self):
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)


class GraphUpsampling(nn.Module):
    """
    Use information from the agglomerative hierarchical clustering for
    doing the upsampling by creating the upsampling matrix U
    """
    def __init__(self, U, A, gamma=0.5, method=WEI):
        # NOTE: gamma = 1 is equivalent to no_A
        super(GraphUpsampling, self).__init__()
        if A is not None:
            assert np.allclose(A, A.T), ERR_A_NON_SYM
            assert method in [BIN, WEI], ERR_WRONG_METHOD
            self.A = np.linalg.inv(np.diag(np.sum(A, 0))).dot(A)
            self.A = gamma*np.eye(A.shape[0]) + (1-gamma)*self.A
            self.A = Tensor(self.A)
            self.U_T = Tensor(U).t().mm(self.A)
        else:
            self.U_T = Tensor(U).t()
        self.parent_size = U.shape[1]
        self.child_size = U.shape[0]
        self.method = method

    def forward(self, input):
        assert input.shape[0] == 1, ERR_WRONG_INPUT_SIZE
        if self.method == REG:
            sf = self.child_size/self.parent_size
            return nn.functional.interpolate(input, scale_factor=sf,
                                             mode='linear',
                                             align_corners=True)
        n_channels = input.shape[1]
        if self.method in [NO_A, BIN, WEI]:
            output = input[0, :, :].mm(self.U_T)
        else:
            raise RuntimeError(ERR_UNK_UPS.format(self.method))
        return output.view([1, n_channels, self.child_size])

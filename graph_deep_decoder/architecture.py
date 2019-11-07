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
        # Decoder Section
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
                # This is not the last layer
                if self.act_fn is not None:
                    self.add_layer(self.act_fn)
                if self.batch_norm:
                    self.add_layer(nn.BatchNorm1d(self.fts[l+1]))
            else:
                # This is the last layer
                if self.last_act_fn is not None:
                    self.add_layer(self.last_act_fn)
        return self.model

    def forward(self, x):
            return self.model(x)

    # To class model??
    def fit(self, signal, mask=None, n_iter=2000, lr=0.01, verbose=False, freq_eval=100):
        p = [x for x in self.model.parameters()]

        optimizer = torch.optim.Adam(p, lr=lr)
        mse = torch.nn.MSELoss()

        if mask is not None:
            mask_var = Tensor(torch.Tensor(mask))

        # It is needed as a torch variable
        signal_var = Tensor(torch.Tensor(signal)).view([1, 1, signal.size])
        best_net = copy.deepcopy(self.model)
        best_mse = 1000000.0

        for i in range(n_iter):
            def closure():
                optimizer.zero_grad()
                out = self.model(self.input)  #.view(signal_var.shape)

                # Choose metric loss depending on the problem
                if mask is not None:  # Inpainting
                    loss = mse(out*mask_var, signal_var*mask_var)
                else:  # Denoising or compression
                    loss = mse(out, signal_var)
                loss.backward()
                return loss

            loss = optimizer.step(closure)

            if verbose and i % freq_eval == 0:
                print('Epoch {}/{}: mse {}'
                      .format(i, n_iter, loss.data))

            if best_mse > 1.005*loss.data:
                best_mse = loss.data
                best_net = copy.deepcopy(self.model)

        self.model = best_net

        return self.model(self.input).detach().numpy(), best_mse

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

        self.parent_size = U.shape[1]
        self.child_size = U.shape[0]
        self.method = method
        self.U_T = Tensor(U).t()

    def forward(self, input):
        assert input.shape[0] == 1, ERR_WRONG_INPUT_SIZE
        if self.method == REG:
            sf = self.child_size/self.parent_size
            return nn.functional.interpolate(input, scale_factor=sf,
                                             mode='linear',
                                             align_corners=True)
        n_channels = input.shape[1]
        output = torch.zeros([1, n_channels, self.child_size])
        if self.method == NO_A:
            output[0, :, :] = input[0, :, :].mm(self.U_T)
        elif self.method in [BIN, WEI]:
            output[0, :, :] = input[0, :, :].mm(self.U_T).mm(self.A)
        else:
            raise RuntimeError('Unknown sampling method')
        return output

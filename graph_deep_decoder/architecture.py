import copy
from enum import Enum

import numpy as np
import torch
from torch import Tensor, eye, manual_seed, nn, no_grad, optim

# from graph_deep_decoder.graph_clustering import Ups


# Constants for Upsampling Type
class Ups(Enum):
    NONE = 0
    LI = 1      # Linear Interpolation
    COPY = 2    # U?     # Directly applies matrix U
    STD = 3     # Applies matrix U and mix each node value with their mean
    GL = 4      # Gamma is a learnable parammeter
    GF = 5
    NVGF = 6


# Error messages
ERR_DIFF_N_LAYERS = 'Length of the nodes and features vector must be the same'
ERR_A_NON_SYM = 'Matrix A for upsampling should be symmetric'
# ERR_WRONG_METHOD = 'Wrong combination of methods for upsampling'
ERR_WRONG_INPUT_SIZE = 'Number of input samples must be 1'
ERR_UNK_UPS = 'Unkown upsampling type {}'
ERR_NO_UPS = 'Upsampling is None but layers have different sizes'


# TODO: ups should be the A_type and ups the type of upsampling op
class GraphDeepDecoder(nn.Module):
    def __init__(self,
                 # Decoder args
                 features, nodes, Us,
                 # Ups args
                 As=None,  ups=Ups.STD,
                 gamma=0.5,
                 batch_norm=True,
                 # Activation functions
                 act_fn=nn.Tanh(), last_act_fn=nn.Tanh()):
        assert len(features) == len(nodes), ERR_DIFF_N_LAYERS
        assert isinstance(ups, Ups) or ups is None, \
            ERR_UNK_UPS.format(ups)

        super(GraphDeepDecoder, self).__init__()
        self.model = nn.Sequential()
        self.fts = features
        self.nodes = nodes
        self.Us = Us
        self.As = As
        self.ups_type = ups
        self.kernel_size = 1
        self.gamma = gamma
        self.batch_norm = batch_norm
        self.act_fn = act_fn
        self.last_act_fn = last_act_fn
        self.build_network()

        if self.ups_type is None or self.ups_type is Ups.NONE:
            shape = [1, self.fts[0], self.nodes[-1]]
        else:
            shape = [1, self.fts[0], self.nodes[0]]
        self.input = Tensor(torch.zeros(shape)).data.normal_()

    def add_layer(self, module):
        self.model.add_module(str(len(self.model) + 1), module)

    def build_network(self):
        ups_skip = 0
        for l in range(len(self.fts)-1):
            self.add_layer(nn.Conv1d(self.fts[l], self.fts[l+1],
                           self.kernel_size, bias=False))

            # Check if there is upsampling in layer l
            if self.nodes[l] < self.nodes[l+1]:
                assert self.ups_type is not (None or Ups.NONE), ERR_NO_UPS
                A = self.As[l+1-ups_skip] if self.As else None

                # Upsampling constructor (?)
                if self.ups_type is Ups.LI:
                    ups = Interpolate(self.nodes[l], self.nodes[l+1])
                elif self.ups_type is Ups.COPY:
                    ups = Upsampling(Us[l-ups_skip])
                elif self.ups_type is Ups.STD:
                    ups = StandardUpsampling(self.Us[l-ups_skip], A,
                                             self.gamma)
                elif self.ups_type is Ups.GL:
                    ups = GLUpsampling(self.Us[l-ups_skip], A,
                                       self.gamma)
                else:
                    raise RuntimeError('Unknown upsampling type')

                self.add_layer(ups)
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
        return sum(p.numel() for p in self.model.parameters()
                   if p.requires_grad)


class Upsampling(nn.Module):
    # TODO: add static method for creating each class of Upsampling
    @staticmethod
    def create(ups_type, U, child_size, parent_size, A, gamma, interp=False):
        if ups_type is Ups.NONE or ups_type is None:
            return None
        elif ups_type is Ups.LI:
            return Interpolate(child_size, parent_size)
        elif ups_type is Ups.COPY:
            return Upsampling(U)
        elif ups_type is Ups.STD:
            return StandardUpsampling(U, A, gamma)
        elif ups_type is Ups.GL:
            ups = GLUpsampling(U, A, gamma)
        else:
            raise RuntimeError('Unknown upsampling type')
        pass

    # NOTE: Previous Ups.NO_A
    def __init__(self, U, interp=False):
        nn.Module.__init__(self)
        if not interp:
            self.child_size = U.shape[0]
            self.parent_size = U.shape[1]
            self.U_T = Tensor(U).t()

    def forward(self, input):
        assert input.shape[0] == 1, ERR_WRONG_INPUT_SIZE
        n_channels = input.shape[1]
        output = input[0, :, :].mm(self.U_T)
        return output.view([1, n_channels, self.child_size])


# NOTE: Previous Ups.REG
class Interpolate(Upsampling):
    def __init__(self, child_size, parent_size):
        Upsampling.__init__(self, [], interp=True)
        self.sf = child_size/parent_size

    def forward(self, input):
        assert input.shape[0] == 1, ERR_WRONG_INPUT_SIZE
        return nn.functional.interpolate(input, scale_factor=self.sf,
                                         mode='linear',
                                         align_corners=True)


# TODO: rethink name!
class StandardUpsampling(Upsampling):
    # NOTE: Previous Ups.BIN/WEI
    def __init__(self, U, A, gamma=0.5):
        assert A is not None
        assert np.allclose(A, A.T), ERR_A_NON_SYM
        Upsampling.__init__(self, U)
        self.A = np.linalg.inv(np.diag(np.sum(A, 0))).dot(A)
        self.A = gamma*np.eye(A.shape[0]) + (1-gamma)*self.A
        self.A = Tensor(self.A)
        self.U_T = Tensor(U).t().mm(self.A.t())


class GLUpsampling(Upsampling):
    # NOTE: previous Learning Gamma
    def __init__(self, U, A, gamma_init=0.5):
        assert A is not None
        assert np.allclose(A, A.T), ERR_A_NON_SYM
        Upsampling.__init__(self, U)
        self.gamma = nn.Parameter(Tensor([gamma_init]))
        self.N = A.shape[0]
        self.A = Tensor(np.linalg.inv(np.diag(np.sum(A, 0))).dot(A))

    def forward(self, input):
        assert input.shape[0] == 1, ERR_WRONG_INPUT_SIZE
        # NOTE: restrict gamma to [0,1] interval?
        n_channels = input.shape[1]
        A_gamma = self.gamma*eye(self.N) + (1-self.gamma)*self.A
        output = input[0, :, :].mm(self.U_T.mm(A_gamma.t()))
        return output.view([1, n_channels, self.child_size])


# class GFUpsampling(nn.Module):
#     def __init__(self, U, A, K):
#         assert A is not None
#         assert np.allclose(A, A.T), ERR_A_NON_SYM
#         super(GFUpsampling, self).__init__()


# ORIGINAL
# class GraphUpsampling(nn.Module):
#     """
#     Use information from the agglomerative hierarchical clustering for
#     doing the upsampling by creating the upsampling matrix U
#     """
#     def __init__(self, U, A, gamma=0.5, method=Ups.WEI):
#         # NOTE: gamma = 1 is equivalent to no_A
#         super(GraphUpsampling, self).__init__()
#         if A is not None:
#             assert np.allclose(A, A.T), ERR_A_NON_SYM
#             assert method in [Ups.BIN, Ups.WEI], ERR_WRONG_METHOD
#             self.A = np.linalg.inv(np.diag(np.sum(A, 0))).dot(A)
#             self.A = gamma*np.eye(A.shape[0]) + (1-gamma)*self.A
#             self.A = Tensor(self.A)
#             self.U_T = Tensor(U).t().mm(self.A.t())
#         else:
#             self.U_T = Tensor(U).t()

#         self.parent_size = U.shape[1]
#         self.child_size = U.shape[0]
#         self.method = method

#     def forward(self, input):
#         assert input.shape[0] == 1, ERR_WRONG_INPUT_SIZE
#         if self.method == Ups.LI:
#             sf = self.child_size/self.parent_size
#             return nn.functional.interpolate(input, scale_factor=sf,
#                                              mode='linear',
#                                              align_corners=True)
#         n_channels = input.shape[1]
#         if self.method in [Ups.NO_A, Ups.BIN, Ups.WEI]:
#             output = input[0, :, :].mm(self.U_T)
#         else:
#             raise RuntimeError(ERR_UNK_UPS.format(self.method))
#         return output.view([1, n_channels, self.child_size])


# TIPOS DE GraphUpsamplings:
#   - Standard
#   - Aprendiendo gamma (GLUpsampling)
#   - Graph Filter
#   - Node-Variant Graph Filter

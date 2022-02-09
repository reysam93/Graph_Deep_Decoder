import copy
from enum import Enum
import math

import numpy as np
import torch
from torch import Tensor, eye, nn
# from torch import Tensor, eye, manual_seed, nn, no_grad, optim

# from graph_deep_decoder.graph_clustering import Ups


# Constants for Upsampling Type
class Ups(Enum):
    NONE = 0
    LI = 1          # Linear Interpolation
    U_MAT = 2       # Directly applies matrix U
    U_MEAN = 3     # Applies matrix U and compute the 1 hop mean
    U_LM = 4        # Learning Mean 1 hope
    GF = 5          # Applies U and then learn a Graph Filter
    NVGF = 6        # Applies U and then learn a NodeVarian-GF


# Error messages
ERR_DIFF_N_LAYERS = 'Length of the nodes and features vector must be the same'
ERR_A_NON_SYM = 'Matrix A for upsampling should be symmetric'
ERR_WRONG_INPUT_SIZE = 'Number of input samples must be 1'
ERR_UNK_UPS = 'Unkown upsampling type {}'
ERR_NO_UPS = 'Upsampling is None but layers have different sizes'


class GraphDecoder(nn.Module):
    """
    This class represent a basic graph decoder with only one layer following
    the model G(C) = ReLU(HC)v, where instead of upsampling a fixed
    low pass graph filter is used.
    """
    def __init__(self, features, H, scale_std=0.01, device='cpu'):
        """
        The arguments are:
        - features: the number of features
        - H: fixed graph filter
        - scale_std: scale the std of the learnable weights initialization
        """
        super().__init__()
        N = H.shape[0]
        self.input = Tensor(H).view([1, N, N])
        self.v = torch.ones(features)/math.sqrt(features)
        self.v[math.ceil(features/2):] *= -1
        self.conv = nn.Conv1d(N, features, kernel_size=1,
                              bias=False)
        std = scale_std/math.sqrt(N)
        self.conv.weight.data.normal_(0, std)
        self.relu = torch.nn.ReLU()

        self.model.to(device)

    def forward(self, input):
        return self.relu(self.conv(input)).squeeze().t().mv(self.v)

    def analytical_squared_jacobian(self):
        """
        Return its squared Jacobian computed analitically
        """
        U = self.input.numpy().squeeze()
        C = self.conv.weight.data.numpy().squeeze().T
        diag_v = np.diag(self.v.numpy().squeeze())
        return (U.dot(C) > 0).dot(diag_v**2).dot((U.dot(C) > 0).T)*(U.dot(U.T))

    def autograd_squared_jacobian(self):
        """
        Return its squared Jacobian computed using autograd
        """
        dec_copy = copy.deepcopy(self)
        y = dec_copy(dec_copy.input)
        outs = torch.eye(y.shape[0])

        jac = []
        for out in outs:
            y = dec_copy(dec_copy.input)
            dec_copy.zero_grad()
            y.backward(out[:])
            allgrads = np.zeros(0)
            for p in dec_copy.parameters():
                gra = p.grad.data.numpy().flatten()
                allgrads = np.append(allgrads, gra)
            jac += [allgrads]
        Jac = np.array(jac).squeeze()
        return Jac.dot(Jac.T)

    def count_params(self):
        return sum(p.numel() for p in self.model.parameters()
                   if p.requires_grad)


class GraphDeepDecoder(nn.Module):
    def __init__(self,
                 # Decoder args
                 features, nodes, Us,
                 # Ups args
                 As=None,  ups=Ups.U_MEAN,
                 gamma=0.5, K=2,
                 batch_norm=False,
                 # Activation functions
                 act_fn=nn.Tanh(), last_act_fn=nn.Tanh(),
                 input_std=0.01, w_std=0.01,
                 device='cpu'):
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
        self.K = K
        self.batch_norm = batch_norm
        self.act_fn = act_fn
        self.last_act_fn = last_act_fn
        self.w_std = w_std/np.sqrt(features)
        self.dev = device
        self.build_network()

        if self.ups_type is None or self.ups_type is Ups.NONE:
            shape = [1, self.fts[0], self.nodes[-1]]
        else:
            shape = [1, self.fts[0], self.nodes[0]]
        std = input_std/np.sqrt(shape[2])
        self.input = Tensor(torch.zeros(shape)).data.normal_(0, std)
        self.input = self.input.to(device)

        self.model.to(device)

    def add_layer(self, module):
        self.model.add_module(str(len(self.model) + 1), module)

    def build_network(self):
        ups_skip = 0
        for l in range(len(self.fts)-1):
            conv = nn.Conv1d(self.fts[l], self.fts[l+1], self.kernel_size,
                             bias=False)
            conv.weight.data.normal_(0, self.w_std[l])
            self.add_layer(conv)

            # Check if there is upsampling in layer l
            if self.Us[l] is not None:
                A = self.As[l] if self.As else None

                if self.ups_type is Ups.LI:
                    ups = Interpolate(self.nodes[l], self.nodes[l+1], self.dev)
                elif self.ups_type is Ups.U_MAT:
                    ups = Upsampling(self.Us[l], self.dev)
                elif self.ups_type is Ups.U_MEAN:
                    ups = MeanUps(self.Us[l], A,
                                  self.gamma, self.dev)
                elif self.ups_type is Ups.U_LM:
                    ups = LearnMeanUps(self.Us[l], A,
                                       self.gamma, self.dev)
                elif self.ups_type is Ups.GF:
                    ups = GFUps(self.Us[l], A, self.K, self.dev)
                elif self.ups_type is Ups.NVGF:
                    ups = NVGFUps(self.Us[l], A, self.K, self.dev)
                else:
                    raise RuntimeError('Unknown upsampling type')

                self.add_layer(ups)

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
        return self.model(x).T.squeeze()

    def count_params(self):
        return sum(p.numel() for p in self.model.parameters()
                   if p.requires_grad)

    def autograd_squared_jacobian(self):
        """
        Return its squared Jacobian computed using autograd
        """
        dec_copy = copy.deepcopy(self)
        y = dec_copy(dec_copy.input)
        outs = torch.eye(y.shape[0])

        jac = []
        for out in outs:
            y = dec_copy(dec_copy.input)
            dec_copy.zero_grad()

            y.backward(out[:])
            allgrads = np.zeros(0)
            for p in dec_copy.parameters():
                gra = p.grad.data.numpy().flatten()
                allgrads = np.append(allgrads, gra)
            jac += [np.array(allgrads)]

        Jac = np.array(jac).squeeze()
        return Jac.dot(Jac.T)

    def Us_norms(self):
        norms = []
        for i, layer in enumerate(self.model):
            if not isinstance(layer, Upsampling):
                continue
            norm = {'layer': i, 'norm': np.linalg.norm(layer.U_T)}
            norms.append(norm)
        return norms

    def Us_to_unit_norm(self):
        for i, layer in enumerate(self.model):
            if not isinstance(layer, Upsampling):
                continue
            layer.U_T /= np.linalg.norm(layer.U_T)


class Upsampling(nn.Module):
    # NOTE: Previous Ups.NO_A
    def __init__(self, U, device='cpu'):
        nn.Module.__init__(self)
        self.child_size = U.shape[0]
        self.parent_size = U.shape[1]
        self.U_T = Tensor(U).t().to(device)

    def forward(self, input):
        assert input.shape[0] == 1, ERR_WRONG_INPUT_SIZE
        # n_channels = input.shape[1]
        # output = input[0, :, :].mm(self.U_T)
        # return output.view([1, n_channels, self.child_size])

        return input.matmul(self.U_T)


# NOTE: Previous Ups.REG
class Interpolate(nn.Module):
    def __init__(self, child_size, parent_size):
        super(Interpolate, self).__init__()
        self.sf = parent_size/child_size

    def forward(self, input):
        assert input.shape[0] == 1, ERR_WRONG_INPUT_SIZE
        return nn.functional.interpolate(input, scale_factor=self.sf,
                                         mode='linear',
                                         align_corners=True)


class MeanUps(Upsampling):
    # NOTE: Previous Ups.BIN/WEI
    def __init__(self, U, A, gamma=0.5, device='cpu'):
        assert A is not None
        assert np.allclose(A, A.T), ERR_A_NON_SYM
        Upsampling.__init__(self, U)
        # NOTE: Not clear if it is better to normalize A!!
        if np.linalg.matrix_rank(np.diag(np.sum(A, 0))) != A.shape[0]:
            self.A = np.linalg.pinv(np.diag(np.sum(A, 0))).dot(A)
            print('WARNING: degree matrix is singular. N:', A.shape[0])
        else:
            self.A = np.linalg.inv(np.diag(np.sum(A, 0))).dot(A)

        # self.A = A
        self.A = gamma*np.eye(A.shape[0]) + (1-gamma)*self.A
        self.A = Tensor(self.A)  #.to(device)
        self.U_T = Tensor(U).t().mm(self.A.t()).to(device)


class LearnMeanUps(Upsampling):
    # NOTE: previous Learning Gamma
    def __init__(self, U, A, gamma_init=0.5, device='cpu'):
        assert A is not None
        assert np.allclose(A, A.T), ERR_A_NON_SYM
        Upsampling.__init__(self, U)
        self.gamma = nn.Parameter(Tensor([gamma_init])).to(device)
        self.A = Tensor(np.linalg.inv(np.diag(np.sum(A, 0))).dot(A)).to(device)

    def forward(self, input):
        assert input.shape[0] == 1, ERR_WRONG_INPUT_SIZE
        # NOTE: restrict gamma to [0,1] interval?
        n_channels = input.shape[1]
        A_gamma = self.gamma*eye(self.child_size) + (1-self.gamma)*self.A
        output = input[0, :, :].mm(self.U_T.mm(A_gamma.t()))
        return output.view([1, n_channels, self.child_size])


class GFUps(Upsampling):
    def __init__(self, U, A, K, device='cpu'):
        assert A is not None
        assert np.allclose(A, A.T), ERR_A_NON_SYM
        super(GFUps, self).__init__(U)
        N = A.shape[0]
        self.hs = nn.Parameter(torch.Tensor(K)).to(device)
        stdv = 1. / math.sqrt(K)
        self.hs.data.uniform_(-stdv, stdv)
        self.A = Tensor(A)
        self.K = K
        self.Apows = torch.zeros(K, N, N).to(device)
        for i in range(K):
            self.Apows[i, :, :] = torch.matrix_power(self.A, i)

    def forward(self, input):
        assert input.shape[0] == 1, ERR_WRONG_INPUT_SIZE
        n_channels = input.shape[1]
        H = (self.hs.view([self.K, 1, 1])*self.Apows).sum(0)
        output = input[0, :, :].mm(self.U_T.mm(H))
        return output.view([1, n_channels, self.child_size])


class NVGFUps(Upsampling):
    def __init__(self, U, A, K):
        assert A is not None
        assert np.allclose(A, A.T), ERR_A_NON_SYM
        super(NVGFUps, self).__init__(U)
        N = A.shape[0]
        self.A = Tensor(A)
        self.hs = nn.Parameter(torch.Tensor(K, N))
        # stdv = 1. / math.sqrt(K)
        # self.hs.data.uniform_(-stdv, stdv)
        self.Apows = torch.zeros(K, N, N)
        self.K = K
        for i in range(K):
            self.Apows[i, :, :] = torch.matrix_power(self.A, i)

    def forward(self, input):
        assert input.shape[0] == 1, ERR_WRONG_INPUT_SIZE
        n_channels = input.shape[1]
        H = (self.hs.view([self.K, self.child_size, 1])*self.Apows).sum(0)
        output = input[0, :, :].mm(self.U_T.mm(H.t()))
        return output.view([1, n_channels, self.child_size])

import copy
import torch
from torch.autograd import Variable
import torch.nn as nn
import numpy as np

# TODO: maybe things like upmethod should use different classes!
class GraphDeepDecoder():
    def __init__(self, descendance, 
                        hier_A, 
                        n_clust,
                        up_method='weighted',
                        n_channels=[5]*3,
                        act_fun=nn.ReLU(),
                        last_act_fun=nn.Sigmoid(),
                        upsampling=True):
        # BUG: check dimmensions --> carefull if A is not used
        # assert(len(descendance)==len(hier_A)-1==len(n_clust)-1==len(n_channels))
        self.descendance = descendance
        self.hier_A = hier_A
        self.n_channels = n_channels + [n_channels[-1]]*2
        self.model = nn.Sequential()
        self.kernel_size = 1
        self.act_fun = act_fun
        self.last_act_fun = last_act_fun
        self.n_clust = n_clust
        self.up_method = up_method
        self.upsampling = upsampling
        if self.upsampling:
            shape = [1, self.n_channels[0], self.n_clust[0]]
        else:
            shape = [1, self.n_channels[0], self.n_clust[-1]]
        self.input = Variable(torch.zeros(shape)).data.normal_()
        #self.input = Variable(torch.zeros(shape)).data.uniform_()

    # NOTE: Equivalent to original arquitecture
    def build_network(self):
        for l in range(len(self.n_channels)-1):
            conv = nn.Conv1d(self.n_channels[l], self.n_channels[l+1], 
                        self.kernel_size, bias=False)
            self.add_layer(conv)

            if l < len(self.n_channels)-2 and self.upsampling:
                self.add_layer(GraphUpsampling(self.descendance[l], self.n_clust[l],
                                                self.hier_A[l+1], self.up_method))
              
            if self.act_fun != None:
                self.add_layer(self.act_fun)
                self.add_layer(nn.BatchNorm1d(self.n_channels[l+1]))
            

        self.add_layer(nn.Conv1d(self.n_channels[-1], 1, 
                        self.kernel_size, bias=False))

        if self.last_act_fun != None:
            self.add_layer(self.last_act_fun)
        return self.model

    def add_layer(self, module):
        self.model.add_module(str(len(self.model) + 1), module)

    def fit(self, signal, n_iter=2000):
        p = [x for x in self.model.parameters() ]

        # torch.optim.Adam(p, lr=0.01)
        optimizer = torch.optim.Adam(p, lr=0.01)
        mse = torch.nn.MSELoss()
        
        # It is needed as a torch variable
        signal_var = Variable(torch.Tensor(signal))
        best_net = copy.deepcopy(self.model)
        best_mse = 1000000.0

        for i in range(n_iter):
            def closure():
                optimizer.zero_grad()
                out = self.model(self.input)
                loss = mse(out, signal_var)
                loss.backward()
                
                #if i % 50 == 0:
                #    out2 = self.model(Variable(self.input))
                #    loss2 = mse(out2, signal_var)
                #    print ('Iteration %05d    Train loss %f  Actual loss %f' % (i, loss.data,loss2.data), '\r', end='')
                return loss

            loss = optimizer.step(closure)

            if best_mse > 1.005*loss.data:
                best_mse = loss.data
                best_net =  copy.deepcopy(self.model)

        self.model = best_net
        print("BEST MSE: ", best_mse)
        return self.model(self.input).detach().numpy(), best_mse

"""
Use information from the agglomerative hierarchical clustering for doing the upsampling by
creating the upsampling matrix U
"""
class GraphUpsampling(nn.Module):
    def __init__(self, descendance, parent_size, A, method):
        super(GraphUpsampling, self).__init__()
        self.descendance = descendance
        self.parent_size = parent_size
        self.A = A
        self.method = method
        self.child_size = len(descendance)
        self.create_U()

    def create_U(self):
        self.U = np.zeros((self.child_size, self.parent_size))
        for i in range(self.child_size):
            self.U[i,self.descendance[i]-1] = 1
        self.U = torch.Tensor(self.U)#.to_sparse()

    def forward(self, input):
        # NOTE:CHECK IF USING INFO OF ITS NEIGHBOURS ONLY OR ALSO ITS OWN!
        n_channels = input.shape[1]
        matrix_in = input.view(self.parent_size, n_channels)

        if self.method == 'no_A':
            output =  self.U.mm(matrix_in)
        elif self.method == 'binary':
            tensor_A = torch.Tensor(self.A/np.sum(self.A,0))
            output =  tensor_A.mm(self.U.mm(matrix_in)) 
        elif self.method == 'weighted':
            tensor_A =  torch.Tensor(self.A)
            output =  tensor_A.mm(self.U.mm(matrix_in)) 
        elif self.method == 'original':
            sf = self.child_size/self.parent_size
            output = torch.nn.functional.interpolate(input, scale_factor=sf,
                                    mode='linear', align_corners=True)
        else:
            raise RuntimeError('Unknown sampling method')

        return output.view(1, n_channels, self.child_size)



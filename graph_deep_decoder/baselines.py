from turtle import forward
import numpy as np
import networkx as nx
import torch_geometric as pyg
from torch import nn
from torch import Tensor
import torch


# Implemented a 3layer GCNN following the paper frim Kipf and Welling
class GCNN(nn.Module):
    def __init__(self, fts, A, input, n_convs=3, n_lin=0, act=nn.ReLU(),
                 last_act=None, last_fts=1, device='cpu'):
        super(GCNN, self).__init__()
        N = A.shape[0]
        if len(input.shape) == 1:
            self.input = Tensor(input).reshape((N, 1)).to(device=device)
            inp_fts = 1
        else:
            self.input = Tensor(input).to(device=device)
            inp_fts = input.shape[1]

        # self.model = pyg.nn.Sequential('x, edge_idx, edge_weight', [
        #     (pyg.nn.GCNConv(inp_fts, fts), 'x, edge_idx, edge_weight -> x1'),
        #     act,
        #     (pyg.nn.GCNConv(fts, fts), 'x1, edge_idx, edge_weight -> x2'),
        #     act,
        #     (pyg.nn.GCNConv(fts, last_fts), 'x2, edge_idx, edge_weight -> x3'),
        # ])

        # sparse_adj = pyg.utils.dense_to_sparse(Tensor(A).to(device=device))
        # self.edge_idx = sparse_adj[0]
        # self.edge_weights = sparse_adj[1]
        # self.last_act = last_act
        # self.model.to(device)

        self.act = act
        self.last_act = last_act
        self.convs = nn.ModuleList()
        self.convs.append(pyg.nn.GCNConv(inp_fts, fts))
        for i in range(1, n_convs):
            if (i == n_convs-1) and n_lin == 0:
                self.convs.append(pyg.nn.GCNConv(fts, last_fts))
            else:
                self.convs.append(pyg.nn.GCNConv(fts, fts))

        self.lins = nn.ModuleList()
        for i in range(n_lin):
            if i == n_lin-1:
                self.lins.append(nn.Linear(fts, last_fts, bias=True))
            else:
                self.lins.append(nn.Linear(fts, fts, bias=True))

        sparse_adj = pyg.utils.dense_to_sparse(Tensor(A).to(device=device))
        self.edge_idx = sparse_adj[0]
        self.edge_weights = sparse_adj[1]
        self.convs.to(device)
        self.lins.to(device)

    def forward(self, x):
        # out = self.model(x, self.edge_idx, self.edge_weights)
        # if self.last_act:
        #     out = self.last_act(out)
        # return out.squeeze()

        x_out = x
        for i, conv in enumerate(self.convs):
            x_out = conv(x_out, self.edge_idx, self.edge_weights)
            if len(self.lins) != 0 or i < len(self.convs)-1:
                x_out = self.act(x_out)

        for i, linear in enumerate(self.lins):
            x_out = linear(x_out)
            if i < len(self.lins)-1:
                x_out = self.act(x_out)

        if self.last_act:
            x_out = self.last_act(x_out)
        return x_out.squeeze()

    def count_params(self):
        return sum(p.numel() for p in self.model.parameters()
                   if p.requires_grad)


# Implemented a 1layer GAT
class GAT(nn.Module):
    def __init__(self, fts, A, heads, input, n_at=1, n_lin=2, act=nn.ReLU(),
                 last_act=None, last_fts=1, device='cpu'):
        assert n_at > 0, 'At least 1 attention layer required.'
        super(GAT, self).__init__()
        N = A.shape[0]
        if len(input.shape) == 1:
            self.input = Tensor(input).reshape((N, 1)).to(device=device)
            inp_fts = 1
        else:
            self.input = Tensor(input).to(device=device)
            inp_fts = input.shape[1]

        # self.model = pyg.nn.Sequential('x, edge_idx, edge_w', [
        #     (pyg.nn.GATConv(inp_fts, fts, heads, edge_dim=1),
        #         'x, edge_idx, edge_w -> x1'),
        #     nn.ReLU(),
        #     (pyg.nn.Linear(heads*fts, fts), 'x1 -> x2'),
        #     nn.ReLU(),
        #     (pyg.nn.Linear(fts, last_fts), 'x2 -> x3'),
        # ])

        # sparse_adj = pyg.utils.dense_to_sparse(Tensor(A).to(device=device))
        # self.edge_idx = sparse_adj[0]
        # self.edge_weights = sparse_adj[1]
        # self.last_act = last_act
        # self.model.to(device)

        self.ats = nn.ModuleList()
        if n_at == 1 and n_lin == 0:
            self.ats.append(pyg.nn.GATConv(inp_fts, last_fts, heads,
                                           concat=False, edge_dim=1))
        else:
            self.ats.append(pyg.nn.GATConv(inp_fts, fts, heads, edge_dim=1))
        for i in range(1, n_at):
            if (i == n_at-1) and n_lin == 0:
                self.ats.append(pyg.nn.GATConv(heads*fts, last_fts, heads,
                                               concat=False, edge_dim=1))
            else:
                self.ats.append(pyg.nn.GATConv(heads*fts, fts, heads,
                                               edge_dim=1))

        self.lins = nn.ModuleList()
        if n_lin == 1:
            self.lins.append(nn.Linear(heads*fts, last_fts, bias=True))
        elif n_lin > 1:
            self.lins.append(nn.Linear(heads*fts, fts, bias=True))
        for i in range(1, n_lin):
            if i == n_lin-1:
                self.lins.append(nn.Linear(fts, last_fts, bias=True))
            else:
                self.lins.append(nn.Linear(fts, fts, bias=True))

        sparse_adj = pyg.utils.dense_to_sparse(Tensor(A).to(device=device))
        self.edge_idx = sparse_adj[0]
        self.edge_weights = sparse_adj[1]
        self.act = act
        self.last_act = last_act
        self.ats.to(device)
        self.lins.to(device)

    def forward(self, x):
        # out = self.model(x, self.edge_idx, self.edge_weights)
        # if self.last_act:
        #     out = self.last_act(out)
        # return out.squeeze()

        x_out = x
        for i, at in enumerate(self.ats):
            x_out = at(x_out, self.edge_idx, self.edge_weights)
            if len(self.lins) != 0 or i < len(self.ats)-1:
                x_out = self.act(x_out)

        for i, linear in enumerate(self.lins):
            x_out = linear(x_out)
            if i < len(self.lins)-1:
                x_out = self.act(x_out)

        if self.last_act:
            x_out = self.last_act(x_out)
        return x_out.squeeze()

    def count_params(self):
        return sum(p.numel() for p in self.model.parameters()
                   if p.requires_grad)


# Implemented the Kron-based graph autoencoder
class KronAE(nn.Module):
    def __init__(self, fts, A, r, input, last_act=None, device='cpu'):
        assert (r >= 0) and (r <= 1), 'r must be between 0 and 1.'

        super(KronAE, self).__init__()
        N = A.shape[0]
        Nt = int(np.round(r*N))

        self.last_act = last_act
        self.fts = fts
        self.dev = device
        self.input = Tensor(input).reshape((N, 1)).to(device=device)
        self.init_reduced_L(A, Nt)

        sparse_adj = pyg.utils.dense_to_sparse(Tensor(A).to(device=device))
        self.edges = sparse_adj[0]
        self.edg_weights = sparse_adj[1]

        sparse_adj_red = pyg.utils.dense_to_sparse(Tensor(self.A_red).
                                                   to(device=device))
        self.edges_red = sparse_adj_red[0]
        self.edg_weights_r = sparse_adj_red[1]

        # Create layers
        self.model = nn.ModuleList()
        self.model.append(pyg.nn.GCNConv(1, fts))
        self.model.append(pyg.nn.GCNConv(fts, fts))
        self.model.append(pyg.nn.GCNConv(fts, 1))
        self.relu = nn.ReLU()
        self.FC = nn.Linear(fts, fts).to(device)
        self.model.to(device)

    def init_reduced_L(self, A, Nt):
        node_deg = np.sum(A, axis=0)
        L = np.diag(node_deg) - A

        # Nt nodes with largest degree will be conserved
        ord_idxs = node_deg.argsort()
        idxs_t = ord_idxs[-Nt:]
        idxs_s = ord_idxs[:Nt]

        L_tt = L[idxs_t, :][:, idxs_t]
        L_ss = L[idxs_s, :][:, idxs_s]
        L_ts = L[idxs_t, :][:, idxs_s]

        # Assuming graph is undirected
        L_red = L_tt - L_ts@np.linalg.pinv(L_ss)@L_ts.T
        self.A_red = np.diag(np.diag(L_red)) - L_red
        self.idxs_t = idxs_t

    def forward(self, x):
        X1 = self.relu(self.model[0](x, self.edges, self.edg_weights))
        X1_red = X1[self.idxs_t, :]
        X2 = self.relu(self.FC(X1_red))
        X3 = torch.zeros(x.shape[0], self.fts).to(self.dev)
        X3[self.idxs_t, :] = X2
        X4 = self.relu(self.model[1](X3, self.edges, self.edg_weights))
        out = self.model[2](X4, self.edges, self.edg_weights).squeeze()
        if self.last_act:
            return self.last_act(out)
        return out

    def count_params(self):
        return sum(p.numel() for p in self.model.parameters()
                   if p.requires_grad)


# Common class for graph unrolling denoising algorithms
class GraphUnrollingDen(nn.Module):
    def __init__(self, A, L, p, alpha, input, device):
        super(GraphUnrollingDen, self).__init__()

        lambdas, V = np.linalg.eigh(A)
        A_norm = A/np.abs(lambdas[0])

        self.input = Tensor(input).to(device=device)
        self.alpha = alpha
        self.P = Tensor(V[:,:p]).to(device)  # Input to MLP
        self.dev = device

        self.create_A_pows(A_norm, L)

    def create_A_pows(self, A, L):
        # A = Tensor(A).to(self.dev)
        A_pows = []
        self.idxs_pows = []
        A = Tensor(A).to(self.dev)
        for l in range(L):
            if l == 0:
                # A_pows.append(A)
                A_pows.append(torch.eye(A.shape[0]).to(self.dev))
            else:
                A_pows.append(A_pows[l-1]@A)

            idxs = torch.where(A_pows[l] != 0)
            self.idxs_pows.append(list(zip([0], idxs[1])))

        self.A_pows = A_pows

    def create_MLPs(self, p, fts):
        MLPs = nn.ModuleList()
        for _ in range(len(self.A_pows)):
            MLP = nn.Sequential(nn.Linear(p, fts),
                                nn.ReLU(),
                                nn.Linear(fts, 1))
            MLPs.append(MLP)
        return MLPs

    def create_conv_matrix(self, MLPs):
        # For K=1 and K'=1
        Conv_mat = torch.zeros(self.A_pows[0].shape).to(self.dev)
        for l, A_l in enumerate(self.A_pows):
            for idx in self.idxs_pows[l]:
                input = self.P[idx[0],:] - self.P[idx[1],:]
                Conv_mat[idx] += A_l[idx]*MLPs[l](input)[0]

        return Conv_mat

    def soft_thres(self, X):
        # Torch
        X_th = torch.zeros(X.shape).to(self.dev)
        X_th = torch.where(X > -self.alpha, X - self.alpha, X_th)
        X_th = torch.where(X < -self.alpha, X + self.alpha, X_th)
        return X_th


# Implemented 1 layer Graph Unrolling Trend Filtering
class GUTF(GraphUnrollingDen):
    def __init__(self, fts, A, L, p, alpha, input, device='cpu'):
        super(GUTF, self).__init__(A, L, p, alpha, input, device)

        G = nx.from_numpy_array(np.sqrt(A))
        self.Delta = nx.linalg.graphmatrix.incidence_matrix(G).todense().T
        self.Delta = Tensor(self.Delta).to(device)
        self.MLPs_B = self.create_MLPs(p, fts).to(device)
        self.MLPs_C = self.create_MLPs(p, fts).to(device)

    def forward(self, x):
        # NOTE: recall that inputs are assumed to be vectors
        x0 = torch.zeros(self.input.shape).to(self.dev)
        y1 = self.soft_thres(self.Delta@x0)
        Conv_B = self.create_conv_matrix(self.MLPs_B)
        Conv_C = self.create_conv_matrix(self.MLPs_C)
        # print(np.linalg.norm(Conv_B.cpu().detach().numpy()))
        return Conv_B@x + Conv_C@(self.Delta.T@y1)


# Implemented 1 layer Graph Unrolling Trend Filtering
class GUSC(GraphUnrollingDen):
    def __init__(self, fts, A, L, p, alpha, input, device='cpu'):
        super(GUSC, self).__init__(A, L, p, alpha, input, device)

        self.MLPs_A = self.create_MLPs(p, fts).to(device)
        self.MLPs_B = self.create_MLPs(p, fts).to(device)
        self.MLPs_D = self.create_MLPs(p, fts).to(device)
        self.MLPs_E = self.create_MLPs(p, fts).to(device)

    def forward(self, x):
        # NOTE: recall that inputs are assumed to be vectors
        s0 = torch.zeros(self.input.shape).to(self.dev)
        z0 = torch.zeros(self.input.shape).to(self.dev)

        # Z_b -> d_b=64
        # S_b -> D_b=64

        Conv_A = self.create_conv_matrix(self.MLPs_A)
        Conv_B = self.create_conv_matrix(self.MLPs_B)
        x1 = Conv_A@s0 + Conv_B@x   # Debe ser 64 fts
        Conv_D = self.create_conv_matrix(self.MLPs_A)
        Conv_E = self.create_conv_matrix(self.MLPs_B)
        # z0 is zero so s1 only depends on x1?
        s1 = Conv_D@x1 + Conv_E@z0  # DEBE SER 64 fts
        
        # y1 = self.soft_thres(self.Delta@x0)
        # Conv_B = self.create_conv_matrix(self.MLPs_B)
        # Conv_C = self.create_conv_matrix(self.MLPs_C)
        # out = Conv_B@x + Conv_C@(self.Delta.T@y1)
        # return out


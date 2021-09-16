from torch import optim, no_grad, nn, Tensor
import copy
import time
import numpy as np

from graph_deep_decoder.architecture import GFUps

# Optimizer constans
SGD = 1
ADAM = 0


class Model:
    def __init__(self, arch,
                 learning_rate=0.001, loss_func=nn.MSELoss(reduction='none'),
                 epochs=1000, eval_freq=100, verbose=False,
                 opt=ADAM):
        assert opt in [SGD, ADAM], 'Unknown optimizer type'
        self.arch = arch
        self.loss = loss_func
        self.epochs = epochs
        self.eval_freq = eval_freq
        self.verbose = verbose
        if opt == ADAM:
            self.optim = optim.Adam(self.arch.parameters(), lr=learning_rate)
        else:
            self.optim = optim.SGD(self.arch.parameters(), lr=learning_rate)

    def count_params(self):
        ps = sum(p.numel() for p in self.arch.parameters() if p.requires_grad)
        return ps

    def get_filter_coefs(self):
        filter_coefs = []
        for layer in self.arch.model:
            if isinstance(layer, GFUps):
                filter_coefs.append(layer.hs.detach().numpy())
        return np.array(filter_coefs)

    def fit(self, signal, x=None, reduce_err=True):
        if x is not None:
            x = Tensor(x)
        x_n = Tensor(Tensor(signal))

        best_err = 1000000
        best_net = None
        best_epoch = 0
        train_err = np.zeros((self.epochs, signal.size))
        val_err = np.zeros((self.epochs, signal.size))
        for i in range(1, self.epochs+1):
            t_start = time.time()
            self.arch.zero_grad()

            x_hat = self.arch(self.arch.input)

            loss = self.loss(x_hat, x_n)
            loss_red = loss.mean()

            if best_err > 1.005*loss_red:
                best_epoch = i
                best_err = loss_red
                best_net = copy.deepcopy(self.arch)

            # Evaluate if the model is overfitting noise
            if x is not None:
                with no_grad():
                    eval_loss = self.loss(x_hat, x)
                    val_err[i-1, :] = eval_loss.detach().numpy()

            loss_red.backward()
            self.optim.step()
            train_err[i-1, :] = loss.detach().numpy()
            t = time.time()-t_start

            if self.verbose and i % self.eval_freq == 0:
                err_val_i = np.sum(val_err[i-1, :])
                err_train_i = np.sum(train_err[i-1, :])
                print('Epoch {}/{}({:.4f}s)\tTrain Loss: {:.8f}\tEval: {:.8f}'
                      .format(i, self.epochs, t,  err_train_i, err_val_i))

        self.arch = best_net
        if reduce_err:
            train_err = np.sum(train_err, axis=1)
            val_err = np.sum(val_err, axis=1)
        return train_err, val_err, best_epoch

    def test(self, x):
        x_hat = self.arch(self.arch.input).squeeze()
        node_err = self.loss(x_hat, Tensor(x)).detach().numpy()
        x_hat = x_hat.detach().numpy()
        err = np.sum(node_err)/np.linalg.norm(x)**2
        return np.median(node_err), err


class BLModel:
    def __init__(self, V, coefs):
        self.V = V
        self.k = min(coefs, V.shape[0])
        self.x_k = None

    def count_params(self):
        return self.coefs

    def fit(self, signal):
        self.x_hat = self.V[:, :self.k].dot(self.V[:, :self.k].T.dot(signal))

    def test(self, x):
        err = np.linalg.norm(x-self.x_hat)**2
        err /= np.linalg.norm(x)**2
        node_err = err/x.size
        return node_err, err


class MeanModel:
    def __init__(self, A):
        self.A = A
        self.D_inv = np.diag(1/self.A.sum(axis=1))

    def fit(self, signal):
        self.x_mean = self.D_inv.dot(self.A).dot(signal)

    def test(self, x):
        err = np.sum((self.x_mean-x)**2)/np.linalg.norm(x)**2
        node_err = err/x.size
        return node_err, err

    def count_params(self):
        return 0


# Total Variation/Graph Filtering model
class TVModel:
    def __init__(self, A, alhpa):
        assert np.isreal(A).all(), 'Only real matrices supported'
        eig_vals, _ = np.linalg.eig(A)
        self.A = A/np.max(np.abs(eig_vals))
        self.alpha = alhpa

    def fit(self, signal):
        assert np.isreal(signal).all(), 'Only real signals supported'
        Eye = np.eye(signal.size)
        A_hat = Eye - self.A
        H_A = np.linalg.inv(Eye+self.alpha*A_hat.T.dot(A_hat))
        self.x_hat = np.asarray(H_A.dot(signal))

    def test(self, x):
        err = np.sum((self.x_hat-x)**2)
        err /= np.linalg.norm(x)**2
        node_err = err/x.size
        return node_err, err

    def count_params(self):
        return 0


# Laplazian Regularizer
class LRModel:
    def __init__(self, L, alhpa):
        self.L = L
        self.alpha = alhpa

    def fit(self, signal):
        assert np.isreal(signal).all(), 'Only real signals supported'
        Eye = np.eye(signal.size)
        self.x_hat = np.linalg.inv(Eye+self.alpha*self.L).dot(signal)
        self.x_hat = np.asarray(self.x_hat)

    def test(self, x):
        err = np.sum((self.x_hat-x)**2)
        err /= np.linalg.norm(x)**2
        node_err = err/x.size
        return node_err, err

    def count_params(self):
        return 0


# median filter
class MedianModel:
    def __init__(self, A):
        self.A = A + np.eye(A.shape[0])

    def fit(self, x):
        self.x_hat = np.zeros(x.shape)
        for i in range(self.A.shape[0]):
            neighbours_ind = np.asarray(self.A[i, :] != 0).nonzero()
            neighbours = x[neighbours_ind]
            self.x_hat[i] = np.median(neighbours)

    def test(self, x):
        err = np.sum((self.x_hat-x)**2)
        err /= np.linalg.norm(x)**2
        node_err = err/x.size
        return node_err, err

    def count_params(self):
        return 0


# Inpainting models
class Inpaint(Model):
    def __init__(self, arch, mask,
                 learning_rate=0.001, loss_func=nn.MSELoss(),
                 epochs=1000, eval_freq=100, verbose=False, opt=ADAM):
        Model.__init__(self, arch, learning_rate=learning_rate,
                       loss_func=loss_func, epochs=epochs,
                       eval_freq=eval_freq, verbose=verbose, opt=opt)
        self.mask = Tensor(mask)

    # NOTE: To many code repeated! --> Change to common funct and only
    # implement the call to loss?
    def fit(self, signal, x=None):
        if x is not None:
            x = Tensor(x).view([1, 1, x.size])
        x_n = Tensor(Tensor(signal)).view([1, 1, signal.size])

        best_err = 1000000
        best_net = None
        best_epoch = 0
        train_err = np.zeros(self.epochs)
        val_err = np.zeros(self.epochs)
        for i in range(1, self.epochs+1):
            t_start = time.time()
            self.arch.zero_grad()

            x_hat = self.arch(self.arch.input)
            loss = self.loss(x_hat*self.mask, x_n*self.mask)
            loss.backward()
            self.optim.step()
            train_err[i-1] = loss.detach().numpy()
            t = time.time()-t_start

            if best_err > 1.005*loss.data:
                best_epoch = i
                best_err = loss.data
                best_net = copy.deepcopy(self.arch)

            # Evaluate if the model is overfitting noise
            # x is never used in the training of the model
            if x is not None:
                with no_grad():
                    eval_loss = self.loss(x_hat, x)
                    val_err[i-1] = eval_loss.detach().numpy()

            if self.verbose and i % self.eval_freq == 0:
                print('Epoch {}/{}({:.4f}s)\tTrain Loss: {:.8f}\tEval: {:.8f}'
                      .format(i, self.epochs, t, train_err[i-1], val_err[i-1]))
        self.arch = best_net
        return train_err, val_err, best_epoch

from torch import optim, no_grad, nn, Tensor
import copy
import time
import numpy as np


# Optimizer constans
SGD = 1
ADAM = 0


class Model:
    def __init__(self, arch,
                 learning_rate=0.001, loss_func=nn.MSELoss(),
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
            loss = self.loss(x_hat, x_n)
            loss.backward()
            self.optim.step()
            train_err[i-1] = loss.detach().numpy()
            t = time.time()-t_start

            if best_err > 1.005*loss.data:
                best_epoch = i
                best_mse = loss.data
                best_net = copy.deepcopy(self.arch)

            # Evaluate if the model is overfitting noise
            if x is not None:
                with no_grad():
                    eval_loss = self.loss(x_hat, x)
                    val_err[i-1] = eval_loss.detach().numpy()

            if self.verbose and i % self.eval_freq == 0:
                print('Epoch {}/{}({:.4f}s)\tTrain Loss: {:.8f}\tEval: {:.8f}'
                      .format(i, self.epochs, t, train_err[i-1], val_err[i-1]))
        self.arch = best_net
        return train_err, val_err, best_epoch

    def test(self, x):
        x_hat = self.arch(self.arch.input).squeeze()
        node_err = self.loss(x_hat, Tensor(x)).detach().numpy()
        x_hat = x_hat.detach().numpy()
        err = np.sum((x_hat-x)**2)/np.linalg.norm(x)**2
        return node_err, err


class Inpaint(Model):
    def __init__(self, arch, mask,
                 learning_rate=0.001, loss_func=nn.MSELoss(),
                 epochs=1000, eval_freq=100, verbose=False, opt=ADAM):
        Model.__init__(self, arch, learning_rate=learning_rate,
                       loss_func=loss_func, epochs=epochs,
                       eval_freq=eval_freq, verbose=verbose, opt=opt)
        self.mask = Tensor(mask)

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
                best_mse = loss.data
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


class BLModel:
    def __init__(self, V, coefs, max_coefs=True):
        self.V = V
        self.coefs = min(coefs, V.shape[0])
        self.max_coefs = max_coefs
        self.x_k = None

    def count_params(self):
        return self.coefs

    def fit(self, signal):
        x_f = self.V.T.dot(signal)
        if self.max_coefs:
            max_indexes = np.argsort(-np.abs(x_f))[:self.coefs]
            self.x_k = self.V[:, max_indexes].dot(x_f[max_indexes])
        else:
            self.x_k = self.V[:, 0:self.coefs].dot(x_f[0:self.coefs])

    def test(self, x):
        err = np.sum((self.x_k-x)**2)/np.linalg.norm(x)**2
        node_err = err/x.size
        return node_err, err

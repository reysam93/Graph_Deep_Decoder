from torch import optim, no_grad, nn, Tensor
import copy
import time
import numpy as np


# Optimizer constans
SGD = 1
ADAM = 0


# TODO: maybe implement 2 models from this, one for denoising and other for inpainting
class Model:
    def __init__(self, arch,
                 learning_rate=0.01, decay_rate=0.99, loss_func=nn.MSELoss(),
                 epochs=1000, eval_freq=100, verbose=False,
                 max_non_dec=10, opt=ADAM):
        assert opt in [SGD, ADAM], 'Unknown optimizer type'
        self.arch = arch
        self.loss = loss_func
        self.epochs = epochs
        self.eval_freq = eval_freq
        self.verbose = verbose
        # self.max_non_dec = max_non_dec
        if opt == ADAM:
            self.optim = optim.Adam(self.arch.parameters(), lr=learning_rate)
        else:
            self.optim = optim.SGD(self.arch.parameters(), lr=learning_rate)
        self.scheduler = optim.lr_scheduler.ExponentialLR(self.optim,
                                                          decay_rate)

    def count_params(self):
        return sum(p.numel() for p in self.arch.parameters() if p.requires_grad)

    def fit(self, signal, x=None, mask=None):
        if mask is not None:
            mask_var = Tensor(mask)

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
            if mask is not None:  # Inpainting
                loss = self.loss(x_hat*mask_var, x_n*mask_var)
            else:  # Denoising or compression
                loss = self.loss(x_hat, x_n)

            loss.backward()
            self.optim.step()
            # self.scheduler.step()
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
        self.arch.eval()
        x_hat = self.arch(self.arch.input).squeeze()
        node_err = self.loss(x_hat, Tensor(x)).detach().numpy()
        x_hat = x_hat.detach().numpy()
        err = np.sum((x_hat-x)**2)/np.linalg.norm(x)**2
        return node_err, err

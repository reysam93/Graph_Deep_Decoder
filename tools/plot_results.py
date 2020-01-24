import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import sem, t
from scipy.io import savemat
import sys

sys.path.insert(0, './graph_deep_decoder')
sys.path.insert(0, '.')
from graph_deep_decoder import utils


PATH = './results/'
SAVE = False

class ExpPlotter():
    def __init__(self, file_name, plot_ind=None, noise_ind=None):
        self.exp_type = file_name.split('_')[0]
        if self.exp_type == 'denoise':
            if file_name.find('real') != -1:
                self.exp_type += '_real'
            self.file = PATH + 'denoising' + '/' + file_name + '.npy'
        else:
            self.file = PATH + 'test_' + self.exp_type + '/' + file_name + '.npy'
        
        data = np.load(self.file).item()
        self.N_P = data['N_P']
        self.error = data['error']
        self.experiments = data['EXPERIMENTS']
        self.select_params(data)
        if noise_ind is not None:
            self.N_P = self.N_P[noise_ind]
            self.error = self.error[noise_ind,:,:]
        if plot_ind is not None:
            self.select_slices(plot_ind)
        self.index_legend()
            
        print('Opening file:', self.file)

    def select_slices(self, plot_ind):
        if not isinstance(plot_ind,list):
            plot_ind = [plot_ind]
        exps_aux = []
        leg_aux = [] 
        fmts_aux = []
        err_aux = None
        for ind in plot_ind:
            if err_aux is None:
                err_aux = self.error[:,:,ind]
            else:
                err_aux = np.concatenate((err_aux,self.error[:,:,ind]),axis=2)
            exps_aux += self.experiments[ind]
            leg_aux += self.legend[ind]
            fmts_aux += self.fmts[ind]    
        self.experiments = exps_aux
        self.legend = leg_aux
        self.fmts = fmts_aux
        self.error = err_aux

    def save_mat_file(self):
        file_name = './results/matfiles/' + self.exp_type
        data = {'error': self.error, 'leg': self.legend,
                'n_p': self.N_P, 'fmts': self.fmts}
        savemat(file_name, data)
        print('Saved as:', file_name)


if __name__ == "__main__":
    # file_name = './results/input/input_2019_12_11-16_04' + '.npy' # denoise_real_2019_07_12-04_02
    file_name = './results/denoise_real/denoise_real_2020_01_21-16_14' + '.npy'

    skip = []
    utils.print_from_file(file_name)
    utils.plot_from_file(file_name, skip)

    # noise_ind = None  # slice(0,12,2)
    # # ARCH: [slice(2), slice(3,5)]
    # # UPS: [slice(4), slice(6,7)]
    # # INP: slice(6)
    # exp_ind = None  # [slice(2), slice(3,5)]
    # plotter = ExpPlotter(file_name,exp_ind, noise_ind)
    # plotter.plot_results()
    # if SAVE:
    #     plotter.save_mat_file()

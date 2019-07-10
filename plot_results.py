import numpy as np
import matplotlib.pyplot as plt

PATH = './results/'

class ExpPlotter():
    def __init__(self, file_name, plot_ind=None):
        self.exp_type = file_name.split('_')[0]
        if self.exp_type == 'denoise':
            if file_name.find('real') != -1:
                self.exp_type += '_real'
            self.file = PATH + 'denoising' + '/' + file_name + '.npy'
        else:
            self.file = PATH + 'test_' + self.exp_type + '/' + file_name + '.npy'
        data = np.load(self.file).item()

        if not 'N_P' in data:
            self.N_P = [0, 0.01, .05, .1, .2, .3, .5]
        else:
            self.N_P =  data['N_P']
        
        self.select_params(data)
        if plot_ind is not None:
            self.select_slices(plot_ind)
            
        
        self.error = data[self.err_key]
        print('Opening file:', self.file)

    def plot_results(self):
        plt.figure()
        median_err = np.median(self.error, axis=1)
        std = np.std(self.error, axis=1)
        prctile25 = np.percentile(self.error, 25, axis=1)
        prctile75 = np.percentile(self.error, 75, axis=1)
        print(np.mean(std,0))
        for i in range(len(self.experiments)):
            # fill_between --> otra posible función
            plt.semilogy(self.N_P, median_err[:,i], self.fmts[i])
            #plt.fill_between(self.N_P, prctile25[:,i], prctile75[:,i], alpha=.3)
        plt.xlabel('Noise Power', fontsize=16)
        plt.ylabel('Median MSE', fontsize=16)
        plt.gca().tick_params(labelsize=16)
        plt.grid(True)
        plt.legend(self.legend, prop={'size': 14})
        plt.show()

    def select_slices(self, plot_ind):
        if not isinstance(plot_ind,list):
            plot_ind = [plot_ind]
        exps_aux = []
        leg_aux = [] 
        fmts_aux = []
        for ind in plot_ind:
            exps_aux += self.experiments[ind]
            leg_aux += self.legend[ind]
            fmts_aux += self.fmts[ind]    
        self.experiments = exps_aux
        self.legend = leg_aux
        self.fmts = fmts_aux
    
    def select_params(self, data):
        self.experiments = data['EXPERIMENTS']
        if self.exp_type == 'denoise':
            self.fmts = data['FMTS']
            self.err_key = 'mse'
            self.legend = [str(exp) for exp in self.experiments]
        elif self.exp_type == 'arch':
            self.fmts = ['o-', '^-', 'P-', 'X-', 'v-']
            self.err_key = 'mse_est'
            self.legend = []
            for i,exp in enumerate(self.experiments):
                text = str(exp['n_chans']) + ', ' + str(data['n_params'][i])
                self.legend.append(text)
        elif self.exp_type == 'ups':
            self.fmts = ['o-', '^-', 'P-', 'X-', 'v-', '<-', '>-', 'o--']
            self.err_key = 'error'
            self.legend = [str(exp) for exp in self.experiments]
        elif self.exp_type == 'input':
            self.fmts = ['o-', '^-', 'X-', 'o--', '^--', 'X--', 'o:', '^:', 'X:']
            self.err_key = 'error'
            self.legend = []
            self.experiments = []
            for inp in data['INPUTS']:
                for exp in data['EXPERIMENTS']:
                    text = inp + ', ' + (exp['ups'] if isinstance(exp, dict) else exp)
                    self.legend.append(text)
                    self.experiments.append({'input': inp, 'exp': exp})
        elif self.exp_type == 'denoise_real':
            self.fmts = ['o-', '^-', 'X-']
            self.err_key = 'error'
            self.legend = [(exp['ups'] if isinstance(exp, dict) else exp) for exp in self.experiments]
        else:
            raise RuntimeError('Unknown experiment type')

if __name__ == "__main__":
    file_name = 'ups_2019_07_08-13_55'
    # UPS:  ups_2019_07_08-13_55
    plotter = ExpPlotter(file_name) #[slice(4), slice(6,7)]
    plotter.plot_results()



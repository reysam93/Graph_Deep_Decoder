import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import sem, t
from scipy.io import savemat

PATH = '../results/'
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
        self.N_P =  data['N_P']
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


    # Probar en varios a usar el intervalo de confianza
    # Y ver si varía mucho entre media y mediana
    def plot_results(self):
        median_err = np.median(self.error, axis=1)
        prctile25 = np.percentile(self.error,25,axis=1)
        prctile75 = np.percentile(self.error,75,axis=1)
        std = np.std(self.error, axis=1)
        confidence = .95

        # Confidence interval
        se = sem(self.error, axis=1)
        n = self.error.shape[1]

        # std from mean??
        print('std', np.median(std,0))
        print('diff 75-25', np.mean((prctile75-prctile25)/median_err*100,0))
        h = se * t.ppf((1+confidence)/2,n-1)
        print(np.mean(h,0))

        # Semilogy median error
        plt.figure(figsize=(7.5,6))
        for i in range(len(self.experiments)):
            plt.semilogy(self.N_P, median_err[:,i], self.fmts[i])
        plt.xlabel('Normalized Noise Power', fontsize=16)
        plt.ylabel('Median Error', fontsize=16)
        plt.gca().autoscale(enable=True, axis='x', tight=True)
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
    
    def select_params(self, data):
        if self.exp_type == 'denoise':
            self.fmts = data['FMTS']
            self.legend = [str(exp) for exp in self.experiments]
        elif self.exp_type == 'arch':
            self.fmts = ['o-', '^-', 'P-', 'X-', 'v-', '*-']
            self.legend = []
            for i,exp in enumerate(self.experiments):
                text = '{}, {}'.format(exp['n_chans'], int(data['n_params'][i])) 
                self.legend.append(text)
        elif self.exp_type == 'ups':
            self.fmts = ['o-', '^-', 'P-', 'X-', 'v-', '<-', '>-', 'o--']
            self.legend = []
            self.legend = [str(exp) for i,exp in enumerate(self.experiments)]
        elif self.exp_type == 'input':
            self.fmts = ['o-', '^-', 'X-', 'o--', '^--', 'X--', 'o:', '^:', 'X:']
            self.legend = []
            self.experiments = []
            for inp in data['INPUTS']:
                for exp in data['EXPERIMENTS']:
                    text = inp + ', ' + (exp['ups'] if isinstance(exp, dict) else exp)
                    self.legend.append(text)
                    self.experiments.append({'input': inp, 'exp': exp})
        elif self.exp_type == 'denoise_real':
            self.fmts = ['o-', '^-', 'X-']
            self.legend = [(exp['ups'] if isinstance(exp, dict) else exp) for exp in self.experiments]
        else:
            raise RuntimeError('Unknown experiment type')

    def index_legend(self):
        new_leg = []
        for i,text in enumerate(self.legend):
            new_leg.append('({}). {}'.format(i+1, text))
        self.legend = new_leg

    def save_mat_file(self):
        file_name = './results/matfiles/' + self.exp_type
        data = {'error': self.error, 'leg': self.legend,
                'n_p': self.N_P, 'fmts': self.fmts}
        savemat(file_name, data)
        print('Saved as:', file_name)

if __name__ == "__main__":
    file_name = 'arch_2019_07_15-03_36' # denoise_real_2019_07_12-04_02
    noise_ind = None #slice(0,12,2)
    # ARCH: [slice(2), slice(3,5)]
    # UPS: [slice(4), slice(6,7)]
    # INP: slice(6)
    exp_ind =  [slice(2), slice(3,5)]
    plotter = ExpPlotter(file_name,exp_ind ,noise_ind) 
    plotter.plot_results()
    if SAVE:
        plotter.save_mat_file()



import numpy as np
import matplotlib.pyplot as plt

PATH = './results/'
# file_name = 'denoising/denoise_N_256_2019_07_01-19_48.npy'

class ExpPrinter():
    def __init__(self, file_name):
        self.exp_type = file_name.split('_')[0]
        if file_name.find('p_n') != -1 or file_name.find('pn') != -1:
            self.partial_print = True
        else:
            self.partial_print = False
        if self.exp_type == 'denoise':
            self.file = PATH + 'denoising' + '/' + file_name + '.npy'
        else:
            self.file = PATH + 'test_' + self.exp_type + '/' + file_name + '.npy'
        print('Opening file:', self.file)
        self.data = np.load(self.file).item()        

    def print_summary(self):
        if self.exp_type == 'arch':
            if self.partial_print:
                self.part_print_arch()
            else:
                self.print_arch()
        elif self.exp_type == 'ups':
            if self.partial_print:
                self.part_print_ups()
            else:
                self.print_ups()
        elif self.exp_type == 'denoise':
            self.print_denoise()
        elif self.exp_type == 'graph':
            self.print_graphs()
        elif self.exp_type == 'input':
            self.print_input()
        elif self.exp_type == 'clust':
            self.print_clust()
        elif self.exp_type == 'inpaint':
            self.print_inpainting()
        elif self.exp_type == 'res':
            self.print_ress()
        else:
            print('ERROR: Unkown file type')

    def part_print_arch(self):
        mean_err = np.mean(self.data['mse_est'],axis=0)
        median_err = np.median(self.data['mse_est'],axis=0)
        std = np.std(self.data['mse_est'],axis=0)
        print('N_P', self.data['n_p'])
        print('batch norm:', self.data['batch_norm'])
        print('last act fn:', self.data['last_act_fun'])
        print('L', self.data['L'])
        for i, exp in enumerate(self.data['EXPERIMENTS']):
            print('{}. {}'.format(i+1, exp))
            print('\tMean ERROR: {}\tMedian ERROR: {}\tSTD: {}'.format(mean_err[i], median_err[i], std[i]))

    """
    OLD VERSION!
    def part_print_arch(self):
        mean_err = np.mean(self.data['mse_est'],axis=0)
        median_err = np.median(self.data['mse_est'],axis=0)
        std = np.std(self.data['mse_est'],axis=0)
        print('N_P', self.data['n_p'])
        print('batch norm:', self.data['batch_norm'])
        print('last act fn:', self.data['last_act_fun'])
        print('L', self.data['L'])
        for i, err in enumerate(mean_err):
            print('{}. Channs: {} Params: {}'.format(i+1, self.data['N_CHANS'][i], self.data['N_CLUSTS'][i]))
            print('\tMean ERROR: {}\tMedian ERROR: {}\tSTD: {}'.format(err, median_err[i], std[i]))
    """

    def print_arch(self):
        print('batch norm:', self.data['batch_norm'])
        print('last act fn:', self.data['last_act_fun'])
        print('L', self.data['L'])
        for i, n_p in enumerate(self.data['N_P']):
            print('NOISE: ', n_p)
            mean_err = np.mean(self.data['error'][i,:,:],axis=0)
            median_err = np.median(self.data['error'][i,:,:],axis=0)
            std = np.std(self.data['error'][i,:,:],axis=0)
            for j, exp in enumerate(self.data['EXPERIMENTS']):
                print('{}. {}'.format(j+1, exp))
                print('\tMean ERR: {}\tMedian ERR: {}\tSTD: {}'.format(mean_err[j], median_err[j], std[j]))
            print()


    def part_print_ups(self):
        mean_error = np.mean(self.data['error'],axis=0)
        median_error = np.median(self.data['error'],axis=0)
        std = np.std(self.data['error'],axis=0)
        print('N_P', self.data['n_p'])
        print('batch norm:', self.data['batch_norm'])
        print('last act fn:', self.data['last_act_fun'])
        print('L', self.data['L'])
        for i, error in enumerate(mean_error):
            print('{}. {}:'.format(i+1, self.data['UPSAMPLING'][i]))
            print('\tMean ERROR: {}\tMedian ERROR: {}\tSTD: {}'.format(error, median_error[i], std[i]))

    def print_ups(self):
        print('batch norm:', self.data['batch_norm'])
        print('last act fn:', self.data['last_act_fun'])
        print('L', self.data['L'])
        for i, n_p in enumerate(self.data['N_P']):
            print('NOISE: ', n_p)
            mean_err = np.mean(self.data['error'][i,:,:],axis=0)
            median_err = np.median(self.data['error'][i,:,:],axis=0)
            std = np.std(self.data['error'][i,:,:],axis=0)
            for j, exp in enumerate(self.data['EXPERIMENTS']):
                print('{}. {}'.format(j+1, exp))
                print('\tMean ERR: {}\tMedian ERR: {}\tSTD: {}'.format(mean_err[j], median_err[j], std[j]))
            print()

    def print_denoise(self):
        print('N_SIGNALS:', self.data['mse'].shape)

        if not 'N_P' in self.data:
            self.data['N_P'] = [0, 0.01, .05, .1, .2, .3, .5]
        
        for j, n_p in enumerate(self.data['N_P']):
            print('NOISE: ', n_p)
            mean_error = np.mean(self.data['mse'][j,:,:],axis=0)
            median_error = np.median(self.data['mse'][j,:,:],axis=0)
            for i, exp in enumerate(self.data['EXPERIMENTS']):
                print('\t{}. {}:'.format(i+1, exp))
                print('\t\tMean ERROR: {} Median ERROR: {}'.format(mean_error[i], median_error[i]))
            print()

    def print_graphs(self):
        print('NOISE: ', self.data['n_p'])
        print('UPS:', self.data['up_method'])
        mean_error = np.mean(self.data['mse_est'],axis=0)
        median_error = np.median(self.data['mse_est'],axis=0)
        std = np.std(self.data['mse_est'],axis=0)
        for i, exp in enumerate(self.data['G_PARAMS']):
            print('\t{}. {}:'.format(i+1, exp))
            print('\t\tMean ERROR: {} Median ERROR: {} STD: {}'.format(mean_error[i], median_error[i], std[i]))

    def print_input(self):
        mean_error = np.mean(self.data['mse'],axis=0)
        median_error = np.median(self.data['mse'],axis=0)
        std = np.std(self.data['mse'],axis=0)

        print(self.data['INPUTS'])
        print(self.data['EXPERIMENTS'])

        #print('NOISE: ', self.data['n_p'])
        for i, s_in in enumerate(self.data['INPUTS']):
            for j, exp in enumerate(self.data['EXPERIMENTS']):
                cont = i*len(self.data['EXPERIMENTS'])+j
                print('{}. INPUT: {} EXP: {}'.format(cont+1, s_in, exp))
                print('\tMean MSE: {}\tMedian MSE: {}\tSTD: {}'
                                .format(mean_error[cont], median_error[cont], std[cont]))

    def print_clust(self):
        mean_error = np.mean(self.data['mse_est'],axis=0)
        median_error = np.median(self.data['mse_est'],axis=0)
        print('N_P', self.data['n_p'])
        print('batch norm:', self.data['batch_norm'])
        print('last act fn:', self.data['last_act_fun'])
        for i, error in enumerate(mean_error):
            print('{}. ALG: {}'.format(i+1, self.data['CLUST_ALGS'][i]))
            print('\tMean ERROR: {}\tMedian ERROR: {}'.format(error, median_error[i]))

    def print_inpainting(self):
        mean_error = np.mean(self.data['mse'],axis=0)
        median_error = np.median(self.data['mse'],axis=0)
        std = np.std(self.data['mse'],axis=0)
        print('p_miss: ', self.data['p_miss'])
        print('Clean Error:', np.mean(self.data['clean_err']))
        for i, exp in enumerate(self.data['EXPERIMENTS']):
            print('{}. EXP: {}'.format(i+1, exp))
            print('\tMean MSE: {}\tMedian MSE: {}\t STD: {}'
                        .format(mean_error[i], median_error[i], std[i]))

    def print_ress(self):
        mean_error = np.mean(self.data['mse_est'],axis=0)
        median_error = np.median(self.data['mse_est'],axis=0)
        print('N_P', self.data['n_p'])
        print('batch norm:', self.data['batch_norm'])
        print('last act fn:', self.data['last_act_fun'])
        for i, error in enumerate(mean_error):
            print('{}. RESOLUTIONS: {}'.format(i+1, self.data['RESOLUTIONS'][i]))
            print('\tMean ERROR: {}\tMedian ERROR: {}'.format(error, median_error[i]))

if __name__ == "__main__":
    file_name = 'arch_2019_07_09-01_48'
    printer = ExpPrinter(file_name)
    printer.print_summary()


#self.data = np.load(path + file_name).item()
#print_denoise(self.data)



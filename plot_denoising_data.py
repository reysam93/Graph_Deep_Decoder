import numpy as np
import matplotlib.pyplot as plt

path = 'E:/Documentos/GitHub/Graph_Deep_Decoder/results/'
file_name = 'denoise_1900_01_01-09_06.npy'

N_P = [0, .001, .005, .01, .05, .1, .5]
EXPERIMENTS = [{'ups': 'original', 'arch': [2,2,2], 't': [4,16,64,None], 'gamma': None},
               {'ups': 'no_A', 'arch': [2,2,2], 't': [4,16,64,None], 'gamma': None},
               {'ups': 'weighted', 'arch': [2,2,2], 't': [4,16,64,None], 'gamma': 0},
               {'ups': 'weighted', 'arch': [2,2,2], 't': [4,16,64,None], 'gamma': .75},
               {'ups': 'original', 'arch': [6,6,6], 't': [4,16,64,None], 'gamma': None},
               {'ups': 'no_A', 'arch': [6,6,6], 't': [4,16,64,None], 'gamma': None},
               {'ups': 'weighted', 'arch': [6,6,6], 't': [4,16,64,None], 'gamma': 0},
               {'ups': 'weighted', 'arch': [6,6,6], 't': [4,16,64,None], 'gamma': .75},
               {'ups': 'original', 'arch': [4,4,4,4], 't': [4,16,64,128,None], 'gamma': None},
               {'ups': 'no_A', 'arch': [4,4,4,4], 't': [4,16,64,128,None], 'gamma': None},
               {'ups': 'weighted', 'arch': [4,4,4,4], 't': [4,16,64,128,None], 'gamma': 0},
               {'ups': 'weighted', 'arch': [4,4,4,4], 't': [4,16,64,128,None], 'gamma': .75},]

fmts = ['o-', '^-', '+-', 'x-', 'o--', '^--', '+--', 'x--', 'o:', '^:', '+:', 'x:']

def plot_results(mean_mse):
    plt.figure()
    for i in range(len(EXPERIMENTS)):
        plt.plot(N_P, mean_mse[:,i], fmts[i])
    
    plt.xlabel('Noise Power')
    plt.ylabel('Mean MSE')
    legend = [str(exp) for exp in EXPERIMENTS]
    plt.legend(legend)


mse = np.load(path + file_name)
plot_results(np.mean(mse, axis=1))
plt.show()

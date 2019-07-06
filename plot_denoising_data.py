import numpy as np
import matplotlib.pyplot as plt

path = 'E:/Documentos/GitHub/Graph_Deep_Decoder/results/denoising/'
file_name = 'denoise_N_256_2019_07_01-19_48.npy'
N_P = [0, 0.01, .05, .1, .2, .3, .5]

def plot_results(data, init, end):
    plt.figure()
    mean_mse = np.mean(data['mse'], axis=1)
    for i in range(len(data['EXPERIMENTS'][init:end])):
        plt.semilogy(N_P, mean_mse[:,i], data['FMTS'][init:end][i])
    plt.xlabel('Noise Power')
    plt.ylabel('Mean MSE')
    plt.grid(True)
    plt.title('Mean MSE')
    legend = [str(exp) for exp in data['EXPERIMENTS'][init:end]]
    plt.legend(legend)



data = np.load(path + file_name).item()
plot_results(data, 0, 12)
plt.show()

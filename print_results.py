import numpy as np
import matplotlib.pyplot as plt

path = 'E:/Documentos/GitHub/Graph_Deep_Decoder/results/'
file_name = 'test_ups/ups_pn_0_2019_07_01-11_26.npy'

def print_results(data):
    mean_error = np.mean(data['mse_est'],axis=0)
    print('N_P', data['n_p'])
    print('batch norm:', data['batch_norm'])
    print('last act fn:', data['last_act_fun'])
    print('L', data['L'])
    for i, error in enumerate(mean_error):
        print('{}.\tMean MSE: {}'.format(i+1,error))


data = np.load(path + file_name).item()
print_results(data)
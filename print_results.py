import numpy as np
import matplotlib.pyplot as plt

path = 'E:/Documentos/GitHub/Graph_Deep_Decoder/results/'
file_name = 'test_ups/ups_pn_0.1_2019_07_01-16_28.npy'

def print_results_arch(data):
    mean_error = np.mean(data['mse_est'],axis=0)
    median_error = np.median(data['mse_est'],axis=0)
    print('N_P', data['n_p'])
    print('batch norm:', data['batch_norm'])
    print('last act fn:', data['last_act_fun'])
    print('L', data['L'])
    for i, error in enumerate(mean_error):
        print('{}. {}, Params: {}'.format(i+1, data['N_CHANS'][i], data['n_params'][i]))
        print('\tMean ERROR: {}\tMedian ERROR: {}'.format(error, median_error[i]))

def print_results_ups(data):
    mean_error = np.mean(data['mse_est'],axis=0)
    median_error = np.median(data['mse_est'],axis=0)
    print('N_P', data['n_p'])
    print('batch norm:', data['batch_norm'])
    print('last act fn:', data['last_act_fun'])
    print('L', data['L'])
    for i, error in enumerate(mean_error):
        print('{}. {}:'.format(i+1, data['UPSAMPLING'][i]))
        print('\tMean ERROR: {}\tMedian ERROR: {}'.format(error, median_error[i]))


data = np.load(path + file_name).item()
print_results_ups(data)
# Create adyacency matrix
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt

# # Constants
# Limit temperature in Fº
MAX_TEMP = 140
MIN_TEMP = -20

# Earth radius in km
R_EARTH = 6371

DATASET_PATH = '../dataset/Temperature-GSOD/'

# # Main variables
# Removing all states
# rm_states_regex = 'WA US|OR US|ID US|MT US|CA US|NV US|WY US|UT US|AZ US|NM US\
# |CO US|ND US|SD US|NE US|KS US|OK US|TX US|LA US|AR US|MO US|IA US|MN US|WI US\
# |IL US|IN US|MI US|OH US|KY US|TN US|MS US|AL US|FL US|GA US|SC US|NC US|VA US\
# |WV US|WA US|MD US|DE US|NJ US|PA US|NY US|CT US|RI US|MA US|VT US|NH US|ME US\
# |AK US|HI US'

# # Graph with 62 nodes
# rm_states_regex = 'NM US\
# |CO US|ND US|SD US|NE US|KS US|OK US|TX US|LA US|AR US|MO US|IA US|MN US|WI US\
# |IL US|IN US|MI US|OH US|KY US|TN US|MS US|AL US|FL US|GA US|SC US|NC US|VA US\
# |WV US|WA US|MD US|DE US|NJ US|PA US|NY US|CT US|RI US|MA US|VT US|NH US|ME US\
# |AK US|HI US'

# # Graph with 130 nodes
# rm_states_regex = 'LA US|AR US|MO US|IA US|MN US|WI US\
# |IL US|IN US|MI US|OH US|KY US|TN US|MS US|AL US|FL US|GA US|SC US|NC US|VA US\
# |WV US|WA US|MD US|DE US|NJ US|PA US|NY US|CT US|RI US|MA US|VT US|NH US|ME US\
# |AK US|HI US'

# # Graph with 188 nodes
# rm_states_regex = 'IN US|MI US|OH US|KY US|TN US|MS US|AL US|FL US|GA US|SC US|NC US|VA US\
# |WV US|WA US|MD US|DE US|NJ US|PA US|NY US|CT US|RI US|MA US|VT US|NH US|ME US\
# |AK US|HI US'

# Graph with 253 nodes
# rm_states_regex = 'SC US|NC US|VA US\
# |WV US|WA US|MD US|DE US|NJ US|PA US|NY US|CT US|RI US|MA US|VT US|NH US|ME US\
# |AK US|HI US'

# Non contiguous states only (316 nodes)
rm_states_regex = 'AK US|HI US'

min_temps = 365
n_signals = 365
knn = 8
dir_graph = False
file_name = '../dataset/temperatures2003'
# file_name = '../temperatures2003_3months'
file_name += '_knn' + str(knn)
if dir_graph:
    file_name += '_dir'

start_time = time.time()
df = pd.read_csv(DATASET_PATH + 'daily_temperature_USA_2003.csv')
print('Original dataset shape:', df.shape)

# Removing states
rm_states = df[df.NAME.str.contains(rm_states_regex, regex=True)]
df = df.drop(rm_states.index)
print('Shape after removing non contiguous states:', df.shape)

# Removing stations with missing data
stations = df.drop_duplicates('NAME')
print('Shape of stations:', stations.shape)
for name in stations.NAME:
    if df[df.NAME == name].shape[0] < min_temps:
        df = df[df.NAME != name]
        continue

stations = df.drop_duplicates('NAME')
print('Stations shape after removing:', stations.shape)

# Read temperature signals X
N = stations.shape[0]
X = np.zeros((N, n_signals))
for i, name in enumerate(stations.NAME):
    temps = df[df.NAME == name]
    X[i, :] = temps.TEMP[:n_signals]

assert np.all(X <= MAX_TEMP) and np.all(X >= MIN_TEMP)
print('Shape of X', X.shape)

# Read stations coordinates and convert to radians
Coords = np.zeros((N, 2))
Coords[:, 0] = stations.LONGITUDE.to_numpy()*np.pi/180
Coords[:, 1] = stations.LATITUDE.to_numpy()*np.pi/180

# Coordinates in km
Coords_km = np.zeros((N, 2))
Coords_km[:, 0] = R_EARTH*Coords[:, 0]*np.cos(Coords[:, 1])
Coords_km[:, 1] = R_EARTH*Coords[:, 1]

# For geodesic distance in km
D = np.zeros((N, N))
for i in range(N):
    for j in range(i+1, N):
        D[i, j] = np.linalg.norm(Coords_km[i, :] - Coords_km[j, :])
D = D + D.T

P = np.exp(-D/np.sum(D)*N**2)
P_n = np.sum(P, axis=0)
np.fill_diagonal(D, np.inf)

idx = D.argsort()[:, :knn]
A = np.zeros(D.shape)
for i in range(N):
    A[i, idx[i, :]] = P[i, idx[i, :]]/P_n[idx[i, :]]
    if not dir_graph:
        A[idx[i, :], i] = A[i, idx[i, :]]

A_bin = np.zeros(A.shape)
A_bin[A != 0] = 1
print('Zeros:', np.sum(A == 0))
print('Non Zeros:', np.sum(A != 0))
print('Mean degree of A:', np.mean(np.sum(A_bin, axis=0)))

plt.figure()
plt.imshow(D)
plt.colorbar()
plt.figure()
plt.imshow(A)
plt.colorbar()
plt.figure()
plt.imshow(A_bin)
plt.colorbar()

file_name += '_N' + str(N)
np.savez(file_name, A=A, X=X, Coords=Coords,
         Coords_km=Coords_km, A_bin=A_bin, D=D)
print('File saved as ', file_name)

print('--- {} minutes ---'.format((time.time()-start_time)/60))
plt.show()

import sys
import numpy as np
import matplotlib.pyplot as plt
from pygsp import graphs
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.sparse.csgraph import dijkstra
from scipy import sparse


# Constants
SEED = 10

# To utilis? --> not vissible (if possible)
"""
Distribute contiguous nodes in the same community while assuring that all
communities have (approximately) the same number of nodes.  
"""
def assign_nodes_to_comms(N,k):
    z = np.zeros(N, dtype=np.int)
    leftover_nodes = N % k
    grouped_nodes = 0
    for i in range(k):
        if leftover_nodes > 0:
            n_nodes = np.ceil(N/k).astype(np.int)
            leftover_nodes -= 1
        else:
            n_nodes = np.floor(N/k).astype(np.int)
        z[grouped_nodes:(grouped_nodes+n_nodes)] = i
        grouped_nodes += n_nodes
    return z

# To utils?
def random_sparse_s(G, min_delta=-5, max_delta=5, epsilon=0.1):
    # Create delta values
    k = G.info['comm_sizes'].shape[0]
    step = (max_delta-min_delta)/(k-1)
    delta_means = np.arange(min_delta, max_delta+0.1, step)
    delta_values = np.random.randn(k) + delta_means

    # Randomly assign delta value to comm nodes
    s = np.zeros((G.W.shape[0]))
    for i in range(k):
        comm_nodes, = np.asarray(G.info['node_com']==i).nonzero()
        rand_index = np.random.randint(0,G.info['comm_sizes'][i])
        selected_node = comm_nodes[rand_index]
        s[selected_node] = delta_values[i]
    return s

# To utils?
def random_diffusing_filter(G, L):    
    # TODO: One filter or one per comm??
    hs = np.random.rand(L)
    H = np.zeros(G.W.shape)
    S = G.W.todense()
    for l in range(L):
        H += hs[l]*np.linalg.matrix_power(S,l)
    return H

def eigenvectors_hier_clusetring(G, library):
    # TODO: consider an unknown library
    G.compute_laplacian()
    G.compute_fourier_basis()
    k = G.info['comm_sizes'].shape[0]
    X = G.U[:,1:2]
    if library == 'sklearn':   
        clustering = AgglomerativeClustering(n_clusters=k,connectivity=G.W).fit(X)
        #clustering = AgglomerativeClustering(n_clusters=k).fit(X)
        return clustering.labels_, None
    else: # library == 'scipy':
        Z = linkage(X,'ward')
        return fcluster(Z,k,criterion='maxclust'), Z

def distance_matrix_hier_clustering(G, library):
    # TODO: consider an unknown library
    D = dijkstra(G.W)
    k = G.info['comm_sizes'].shape[0]
    #clustering = AgglomerativeClustering(n_clusters=k,connectivity=G.W,
    #                                affinity='precomputed',linkage='average').fit(D)
    # Aparently better without connectivity?? 0.0
    if library == 'sklearn':
        clustering = AgglomerativeClustering(n_clusters=k,
                                    affinity='precomputed',linkage='average').fit(D)
        return clustering.labels_, None
    else:
        D = D[np.triu_indices_from(D,1)]
        Z = linkage(D,'ward')
        return fcluster(Z,k,criterion='maxclust'), Z

def hierarchical_clustering(G, algorithm, library='scipy', plot_dn=True):
    # Type must be distance or eigenvector
    if algorithm == 'distance':
        labels, Z =  distance_matrix_hier_clustering(G, library)
    elif algorithm == 'eigenvectors':
        labels, Z = eigenvectors_hier_clusetring(G, library)

    if library =='scipy' and plot_dn:
        plt.figure()
        dendrogram(Z)
    return labels

if __name__ == '__main__':
    # SBM graph parameters
    N = 20#100
    k = 3
    p = 0.6#0.4
    q = 0.1#0.01/k
    rand_z = False
    z = None
    L = 3
    alg = 'eigenvectors'
    clust_lib = 'scipy'

    np.random.seed(SEED)

    print("Creating random graph")
    if not rand_z:
        z = assign_nodes_to_comms(N,k)
    try:
        G = graphs.StochasticBlockModel(N=N, k=k, z=z, p=p, q=q,
                                        connected=True, seed=SEED)
    except ValueError as e:
        sys.exit(e)

    print("Creating graph signal by diffusing deltas placed in each comm")
    s = random_sparse_s(G)
    H = random_diffusing_filter(G,L)
    x = np.asarray(H.dot(s))

    print("Clusterizing the graph")
    labels = hierarchical_clustering(G, alg, clust_lib)

    # Plot
    # TODO: use a plot function 
    print(s[s!=0])
    G.set_coordinates(kind='community2D')
    fig, axes = plt.subplots(1, 2)
    _ = axes[0].spy(G.W, markersize=0.8)
    # Grph topology representaiton on second
    G.plot_signal(labels, ax=axes[1])
    plt.show()

    # TODO: choose where to cut the clusters
    # TODO: create graph deep decoder architecture for reconstructing 
    # signal from white noise
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import networkx as nx
import random as rd
import seaborn as sns
from community import community_louvain
from memory_profiler import profile
from time import time
from sklearn.metrics import normalized_mutual_info_score
from networkx.generators.community import LFR_benchmark_graph


## Exercice 1 ##

def generate_cluster_graph(p, q, nb_clusters=4, nb_nodes_c=100):
    '''Generate a graph with nb_clusters clusters, where nodes are connected with different probabilities

    -- Input --
    p (float): probability of connection between nodes of same cluster
    q (float): probability of connection between nodes of different clusters (q <= p)
    nb_clusters (int): number of clusters
    nb_nodes (int): number of nodes within each cluster

    -- Output --
    edge_list (list of tuples): graph stored as an edge list
    '''
    edge_list = []
    node_list = np.arange(nb_nodes_c*nb_clusters)  # Create the list of nodes
    for idx_1, node_1 in enumerate(node_list):
        cluster_idx = idx_1 // nb_nodes_c  # Retrieve the index of the cluster node_1 belongs to
        for node_2 in node_list[idx_1+1:(cluster_idx+1)*nb_nodes_c]:  # If node_2 is in the same cluster as node_1, with node_2 > node_1
            if np.random.random_sample() < p:  # With probability p
                edge_list.append((node_1, node_2))  # Create the edge (node_1, node_2)
        for node_3 in node_list[(cluster_idx+1)*nb_nodes_c:]:  # If node_3 is in a superior cluster to node_1's
            if np.random.random_sample() < q:  # With probability q
                edge_list.append((node_1, node_3))  # Create the edge (node_1, node_3)
    return edge_list


## Exercice 2 ##

def convert_edge_to_adjarray(G):
    '''Convert a graph stored as list of edges to an adjacency array

    -- Input --
    G (list of tuples): graph stored as list of edges

    -- Output --
    G_adj (dict): graph stored as adjacency array
    '''
    G_nx = nx.from_edgelist(G)  # Convert to networkx graph
    G_adj = nx.to_dict_of_dicts(G_nx)  # Convert to dictionary of dictionaries (each node refers to a dict whose keys are its neigbors)
    for keys, values in G_adj.items():
        G_adj[keys] = [i for i in values.keys()]
    return G_adj


def fisher_yates_shuffle(l):
    '''Shuffle array l according to Fisher-Yates algorithm
    '''
    for i in range (len(l)-1, -1, -1):
        j = rd.choice(np.arange(i+1))
        temp = l[j]
        l[j] = l[i]
        l[i] = temp
    return l


def label_propagation(G_adj):
    '''Label propagation algorithm to find communities by giving to each node the label of highest frequency label among its neigbors

    -- Input --
    G (dictionary): graph stored as adjacency dictionary

    -- Output --
    label (list): list of labels indexed by node
    '''
    label = np.arange(len(G_adj.keys()))  # Create label list referencing label of each node
    key_iterator = list(G_adj.keys())  # Create an iterator over the keys which will determine in which order the labels are changed because dictionaries can't be shuffled
    current_diff = 1
    nb_it = 0  # Number of iterations
    while current_diff > 0:  # As long as the previous iteration changed labels
        prev_label = label.copy()  # Copy the list of labels
        current_diff = 0  # Reset
        key_iterator = fisher_yates_shuffle(key_iterator)  # Shuffle the order
        for node in key_iterator:  # Go over every node in the shuffled order
            neighbor_label = [label[neighbor] for neighbor in G_adj[node]]  # Create the list of neighbor labels
            neighbor_label = fisher_yates_shuffle(neighbor_label)  # We also shuffle the neighbor labels as the max method takes the first if all labels occur the same number of times
            label[node] = max(neighbor_label, key=neighbor_label.count)  # Change label of node to label with highest occurrence
            current_diff += abs(label[node] - prev_label[node])  # Add a term if the label has changed
        nb_it += 1
    return label, nb_it


def plot_detected_communities(G_adj, label, show_fig=True):
    '''Plot the graph by coloring the communities detected according to label

    -- Input --
    G_adj (dict): graph stored as an adjacency matrix
    label (list): list of labels indexed by node (output of label_propagation algorithm)
    '''
    # First we convert the graph to networkx graph structure
    G_nx = nx.from_dict_of_lists(G_adj)

    # Create dictionary for different labels
    unique_labels = set(label)
    l = np.arange(len(unique_labels))
    label_dict = dict(zip(unique_labels, l))

    palette = sns.color_palette(None, len(unique_labels))
    color_map = []
    for node in G_nx:
        palette_idx = label_dict[label[node]]
        color_map.append(palette[palette_idx])

    if show_fig:
        nx.draw(G_nx, node_color=color_map, with_labels=False)
        plt.show()

    return unique_labels


def louvain_partition(G_nx, show_fig=True):
    '''Compute partition of the graph nodes using the Louvain heuristics and plot the corresponding colored graph
    '''
    partition = community_louvain.best_partition(G_nx)  # dictionary labeling each node according to its community
    if show_fig:
        cmap = cm.get_cmap('viridis', max(partition.values()) + 1)
        nx.draw(G_nx, cmap=cmap, node_color=list(partition.values()))
        plt.show()
    return


def accuracy_algo(p, q, nb_clusters=4, nb_nodes_c=100, algo='label prop'):
    '''Calculate the accuracy of a community detection algorithm by looking at each arc and checking if both nodes are correctly labeled or not,
    by using it on a generated cluster graph

    -- Input --
    p (float): probability of connection between nodes of same cluster
    q (float): probability of connection between nodes of different clusters (q <= p)
    nb_clusters (int): number of clusters
    nb_nodes (int): number of nodes within each cluster
    algo (str): community detection algorithm to use

    -- Output --
    acc (float): value between 0 and 1, the mean of accuracy of each arc in the graph
    '''
    G = generate_cluster_graph(p, q, nb_clusters, nb_nodes_c)  # We know the communities in this graph
    nb_nodes = nb_clusters*nb_nodes_c  # Total number of nodes
    # Compute the 'true' labels for the nodes in their community
    true_label = [[i]*nb_nodes_c for i in range(1, nb_clusters + 1)]
    true_label = np.array(true_label).flatten()

    if algo == 'label prop':
        G_adj = convert_edge_to_adjarray(G)
        labels, nb_it = label_propagation(G_adj)

    if algo == 'louvain':
        G_nx = nx.from_edgelist(G)
        labels = community_louvain.best_partition(G_nx)

    tp = 0
    tn = 0
    fp = 0
    fn = 0
    for node_1 in range(nb_nodes):
        for node_2 in range(node_1+1, nb_nodes):
            # If the labels of the pair are the same for predicted and actual community
            if labels[node_1] == labels[node_2] and true_label[node_1] == true_label[node_2]:
                tp+=1
            
            # If the labels of the pair are different for predicted and actual community
            if labels[node_1] != labels[node_2] and true_label[node_1] != true_label[node_2]:
                tn+=1
            
            # If the labels of the pair are the same for predicted and different for the actual community
            if labels[node_1] == labels[node_2] and true_label[node_1] != true_label[node_2]:
                fn+=1

            # If the labels of the pair are different for the pair and the same for actual community
            if labels[node_1] != labels[node_2] and true_label[node_1] == true_label[node_2]:
                fp+=1

    return (tp+tn)/(tp+tn+fp+fn)


def plot_accuracy(nb_clusters, nb_nodes_c):
    '''Plot the accuracy of the algorithms for community detection based on the ratio p/q
    '''
    p = 0.1
    q_list = np.arange(0.005, 0.1, 0.005)
    err_list_louvain = []
    err_list_label_prop = []
    ratio_list = []
    for q in q_list:
        err_list_label_prop.append(accuracy_algo(p, q, nb_clusters, nb_nodes_c, algo='label prop'))
        err_list_louvain.append(accuracy_algo(p, q, nb_clusters, nb_nodes_c, algo='louvain'))
        ratio_list.append(p/q)

    plt.plot(ratio_list, err_list_label_prop)
    plt.plot(ratio_list, err_list_louvain)
    plt.title('Accuracy des algorithmes en fonciton du ratio p/q')
    plt.show()


def accuracy_lfr(tau1, tau2, mu, n=300, algo='label prop'):
    '''Calculate the accuracy of the algorithms for community detection based on the lfr benchmark
    '''
    G_nx = LFR_benchmark_graph(n, tau1, tau2, mu, min_degree=3, min_community=50, seed=10)
    communities = list({frozenset(G_nx.nodes[v]["community"]) for v in G_nx})
    label = np.arange(len(communities))
    true_label = {}
    for i in range(n):
        for j in label:
            if i in communities[j]:
                true_label[i] = j

    G = nx.to_edgelist(G_nx)

    if algo == 'label prop':
        G_adj = convert_edge_to_adjarray(G)
        labels, nb_it = label_propagation(G_adj)

    if algo == 'louvain':
        G_nx = nx.from_edgelist(G)
        labels = community_louvain.best_partition(G_nx)

    tp = 0
    tn = 0
    fp = 0
    fn = 0
    for node_1 in range(n):
        for node_2 in range(node_1+1, n):
            # If the labels of the pair are the same for predicted and actual community
            if labels[node_1] == labels[node_2] and true_label[node_1] == true_label[node_2]:
                tp+=1
            
            # If the labels of the pair are different for predicted and actual community
            if labels[node_1] != labels[node_2] and true_label[node_1] != true_label[node_2]:
                tn+=1
            
            # If the labels of the pair are the same for predicted and different for the actual community
            if labels[node_1] == labels[node_2] and true_label[node_1] != true_label[node_2]:
                fn+=1

            # If the labels of the pair are different for the pair and the same for actual community
            if labels[node_1] != labels[node_2] and true_label[node_1] == true_label[node_2]:
                fp+=1

    return (tp+tn)/(tp+tn+fp+fn)


# Main

# Ex 1
# G = generate_cluster_graph(0.5, 0.1, nb_clusters=2, nb_nodes=100)
# G_nx = nx.from_edgelist(G)
# nx.draw(G_nx, with_labels=False)
# plt.show()  # Clearly, when we increase the ratio p/q, we can distinguish the communities, while the closer the ratio is to 1, the harder it becomes

# # Ex 2
# G_adj = convert_edge_to_adjarray(G)
# t0 = time()
# labels, nb_it = label_propagation(G_adj)
# t1 = time()
# a = plot_detected_communities(G_adj, labels, show_fig=True)
# print(a)  # nombre de communautés détectées
# print('Label propagation time :', t1-t0)

# Ex 3

# Scalability
scale = False

if scale:
    G = generate_cluster_graph(0.15, 0.02, nb_clusters=8, nb_nodes_c=2000)
    G_nx = nx.from_edgelist(G)
    print('Number of nodes : ', G_nx.number_of_nodes(), '\nNumber of edges : ', G_nx.number_of_edges())

    G_adj = convert_edge_to_adjarray(G)
    t0 = time()
    labels, nb_it = label_propagation(G_adj)
    t1 = time()
    print('Label propagation time :', t1-t0)
    t2 = time()
    louvain_partition(G_nx, show_fig=False)
    t3 = time()
    print('Louvain partition time :', t3-t2)


# Accuracy
# plot_accuracy(4, 100)

# LFR benchmark
n=250
tau1=3
tau2=1.5
mu=0.1
G_nx = LFR_benchmark_graph(n, tau1, tau2, mu, average_degree=5, min_community=20, seed=10)
nx.draw(G_nx)
plt.show()


print(accuracy_lfr(tau1, tau2, mu, algo='label prop'))

import torch
import networkx as nx
import numpy as np
import random as rd
import matplotlib.pyplot as plt
from collections import Counter
import os
import pandas as pd


"""this function create the initial individual dictionnary from the embedding of all epochs output of the GNN"""
def create_indiv(h_all, G):
    values=[sublist[0] for sublist in h_all[0]]

    indiv={node: val for node, val in zip(G.nodes, values)}
    return indiv


def symmetric_normalized_laplacian(G):
    """
    Build the symmetric normalized Laplacian L_sym from a NetworkX graph G.

    L_sym = I - D^{-1/2} A D^{-1/2}

    Returns:
        L_sym (np.ndarray): (n,n) symmetric normalized Laplacian
        nodelist (list): order of nodes used for the matrix
    """
    nodelist = list(sorted(G.nodes()))
    
    A = nx.to_numpy_array(G, nodelist=nodelist, weight='weight')
    
    d = A.sum(axis=1)
    
    d_inv_sqrt = 1.0 / np.sqrt(np.maximum(d, 1e-12))
    D_inv_sqrt = np.diag(d_inv_sqrt)
    
    L_sym = np.eye(len(nodelist)) - D_inv_sqrt @ A @ D_inv_sqrt
    return L_sym, nodelist


def dirichlet_energy_v2(h, G, add_self_loop=True):
    """
    h: (n, d) numpy array of node embeddings (order must match G.nodes())
    G: networkx.Graph (edge weights in attribute 'weight', defaults to 1)
    add_self_loop: whether to use A+I renormalization (useful for GCN)
    Returns: scalar Dirichlet energy using symmetric normalized Laplacian
             i.e. 0.5 * sum_{ij} A_ij || h_i/sqrt(d_i) - h_j/sqrt(d_j) ||^2
    """

    nodelist = list(sorted(G.nodes()))
    A = nx.to_numpy_array(G, nodelist=nodelist, weight='weight')

    if add_self_loop:
        A = A + np.eye(A.shape[0])

    deg = A.sum(axis=1)
    deg_sqrt_inv = 1.0 / np.sqrt(np.maximum(deg, 1e-12))

    h = np.asarray(h, dtype=float)
    h_tilde = (deg_sqrt_inv[:, None]) * h 

    E = 0.0
    for i, j, data in G.edges(data=True):
        pass

    D_inv_sqrt = np.diag(deg_sqrt_inv)
    L_sym = np.eye(A.shape[0]) - D_inv_sqrt @ A @ D_inv_sqrt
    E_trace = np.trace(h.T @ L_sym @ h)
    return float(E_trace)




"""This function compute the distance (N2) between each node embedding and his closest neighbor,take the minimum of these distances for all nodes, and return the sum of squared distances"""
def compute_distance_v2(mean_dict, G, verbose=True):
    dist = 0.0

    for node in G.nodes():
        neighbors = list(G.neighbors(node))
        if not neighbors:   # isolated node
            continue

        f_node = np.array([d[node] for d in mean_dict])

        neighbor_dists = []
        for neigh in neighbors:
            f_neigh = np.array([d[neigh] for d in mean_dict])
            diff = np.linalg.norm(f_node - f_neigh, ord=2)
            neighbor_dists.append(diff)

        # take min over neighbors
        dist_one_node = min(neighbor_dists)

        if verbose:
            print(f"Distance for node {node}: {dist_one_node}")

        dist += dist_one_node


    dist= dist**2 #for easier math cmp
    return dist





def T2_step(indiv, G, weight):
    """
    Compute T2(x) exactly as:
    T2(x_i) = sum_p a_ip * ( sum_q a_pq x_q * 2*(delta_{i,q}+1) / (sum_q a_pq + 2) )
              / (sum_p a_ip + 1)
    """

    old_indiv = indiv.copy()
    new_indiv = {}

    for i in G.nodes:
        neighbors_i = list(G.neighbors(i))

        sum_a_ip = sum(weight[i][p] for p in neighbors_i)
        outer_denom = sum_a_ip + 1

        outer_sum = 0.0

        for p in neighbors_i:
            neighbors_p = list(G.neighbors(p))

            sum_a_pq = sum(weight[p][q] for q in neighbors_p)
            inner_denom = sum_a_pq + 2

            inner_sum = 0.0
            for q in neighbors_p:
                delta = 1 if q == i else 0
                inner_sum += (
                    weight[p][q]
                    * old_indiv[q]
                    * 2 * (delta + 1)
                )

            outer_sum += weight[i][p] * (inner_sum / inner_denom)

        new_indiv[i] = outer_sum / outer_denom

    return new_indiv




"""This function compute the evolution function allong with T2 steps, it take into arg. the number of step M (not really used anymore, set to 2)
the weight of the graph, the graph and individuals and return all the states at each steps"""
def evolve_v6(indiv,G,weight,M=2, ploting_option=True):

    all_state_cp = []

    all_state_cp.append(indiv.copy())
    for _ in range(M-1):
        indiv = T2_step(indiv, G, weight)
        all_state_cp.append(indiv.copy())

    return all_state_cp


"""This function load the embedding from a specific path, and return all the embeddings for each epoch"""
def load_embeding(num_epoch, batch_size, hidden_dim, num_nodes, path="/home/killian/Documents/Data/htest/", verbose=True):
    h_all=[]
    for k in range(0,num_epoch):
        emb=torch.load(f"{path}embeddings_epoch{k}.pt")
        h = emb.view(batch_size, num_nodes, hidden_dim)
        h_last_batch = h[-1]  # shape: [51, 60]
        if verbose:
            print(f"epoch {k}")
        h_all.append(h_last_batch.numpy())
    return h_all


"""This function compute the evolution for all dimensions of the embedding"""
def compute_for_all_dimensions_v2(hidden_dim, M, G, h_all,weight, epoch_to_consider=0, verbose=True):
    

    set_evolved_dimensions = []
    for dim in range(hidden_dim):
        values = [sublist[dim] for sublist in h_all[epoch_to_consider]]
        indiv = {node: val for node, val in zip(G.nodes, values)}
        if verbose:
            print(indiv)
        set_evolved_dimensions.append(evolve_v6(indiv, G, weight, M=M, ploting_option=verbose)[M-1])

    return set_evolved_dimensions

"""This function create the initial individual dictionnary from the embedding of a specific epoch"""
def build_indiv(G,h, hidden_dim): 
    indiv_all_dim=[]
    for dim in range(hidden_dim):

        values = [sublist[dim] for sublist in h] 
        indiv = {node: val for node, val in zip(G.nodes, values)}
        indiv_all_dim.append(indiv)
    return indiv_all_dim


"""This function transform the individual dictionnary into an array, changing the index order"""
def transform_individuals_to_array(indiv):
    vectors_is_nice=[]
    for i in range(len(indiv[0])):
        one_node_vec=[]
        for j in range(len(indiv)):
            one_node_vec.append(indiv[j][i])
        
        vectors_is_nice.append(one_node_vec)
    return vectors_is_nice

"""This function compute the most pivot nodes, i.e. the nodes that have moved the most between the initial and final embedding"""
def get_most_pivot(indiv_init, indiv_end, top1=10):
    pivot=[]

    indiv_init_array=np.array(transform_individuals_to_array(indiv_init))
    indiv_end_array=np.array(transform_individuals_to_array(indiv_end))

    build_dico_keeper={i:indiv_init_array[i] for i in range(len(indiv_init_array))} #normaly choold be same nodes assiciation in final
    dist_all=[]
    for indiv_k, indiv_val in build_dico_keeper.items():

        dist_all.append(np.linalg.norm(indiv_end_array[indiv_k]-indiv_val))
        
    dist_dictionary={i:dist_all[i] for i in range(len(dist_all))}

    dist_sorted_dictionary = dict(sorted(dist_dictionary.items(), key=lambda item: item[1], reverse=True))
    number_to_keep= int(len(dist_sorted_dictionary) * top1 / 100)

    if number_to_keep > 1:
        pivot = dict(list(dist_sorted_dictionary.items())[:number_to_keep])
    else:
        pivot=[]
        print("warning, not enough nodes to keep for pivot, return empty list")


    return pivot



"""This function compute all the metrics for a specific dataset, loading the embedding from a specific path, but only for the last epoch"""
def compute_last_for_a_dataset(path_embedding,G,weight_matrix, num_epoch, hidden_dim, M, dim=0,top=10, verbose=True):

    h_all=[]

    if num_epoch==0:
        emb=np.load(path_embedding)
    else:
        if dim !=0:
            emb=np.load(f"{path_embedding}embeddings_epoch_{num_epoch-1}_{dim}.npy")
            print(emb.shape)
        else:
            emb=np.load(f"{path_embedding}embeddings_epoch_{num_epoch-1}.npy")
            print(emb.shape)
    h_all.append(emb)
    weight = weight_matrix
    evolved_dim=compute_for_all_dimensions_v2( hidden_dim, M, G,h_all,weight, epoch_to_consider=0, verbose=verbose)
    dist_all=compute_distance_v2(evolved_dim, G,verbose=False)
    indiv_init=build_indiv(G,h_all[0], hidden_dim)
    pivot=get_most_pivot(indiv_init, evolved_dim, top1=top)
    return h_all,evolved_dim,dist_all,pivot



"""This function is similar to the previous one, the only difference is that it works on a previously loaded embedding, there is no more epoch choice here"""
def compute_loaded_for_a_dataset(embedding,G,weight_matrix, hidden_dim, M,top=10, verbose=True):

    h_all=[]
    emb=embedding
    h_all.append(emb)

    weight = weight_matrix
    evolved_dim=compute_for_all_dimensions_v2(hidden_dim, M, G,h_all,weight, epoch_to_consider=0, verbose=verbose)
    dist_all=compute_distance_v2(evolved_dim, G,verbose=False)
    indiv_init=build_indiv(G,h_all[0], hidden_dim)
    pivot=get_most_pivot(indiv_init, evolved_dim, top1=top)
    return h_all,evolved_dim,dist_all,pivot

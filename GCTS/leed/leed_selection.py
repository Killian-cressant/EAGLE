import networkx as nx
from .tools_for_evolution import compute_loaded_for_a_dataset
from torch_geometric.utils import to_networkx
import numpy as np
from gcts.exp_util import preprocess_graph_v2

import os
import pickle

import torch

def evolve_centrality_wrapper(graph,top=100, centrality_cache_path=None):


    G = to_networkx(graph, to_undirected=True)
    edge_index = graph.edge_index
    if graph.x != None:
        embedding_np = graph.x.T.cpu().numpy()
    else:
        embedding_np = np.ones((G.number_of_nodes(), 1))
    

    adjs = graph.A_norm
    hidden_d= len(embedding_np[0])
    _,_,_,pivot=compute_loaded_for_a_dataset(embedding=embedding_np, G=G, weight_matrix=adjs, hidden_dim=hidden_d, M=2, verbose=False, top=top)

    centralities = pivot

    return centralities
    




"""Local leed in time"""
def get_all_graph(loader, args, data_type= 'None', dataset_name=None):

    if data_type is not None and dataset_name is not None:
        filename = f"data/{dataset_name}/centrality_{data_type}.pkl"
        
        if os.path.exists(filename) and os.path.getsize(filename) > 0:
            print(f"Loading cached data from {filename}")
            with open(filename, "rb") as f:
                return pickle.load(f)

    
    centralities_per_window=[]
    
    g=0
    g_previous=None
    centralities_prev=None
    for graph in loader:
        g+=1
        print(g)
        graph = preprocess_graph_v2(graph)

        if g_previous is not None:
            A_prev=g_previous.A
            A_current=graph.A
            #compute distance
            dist=np.linalg.norm(A_prev-A_current, ord='fro')
            print(dist)
            if dist<10:
                centralities=centralities_prev
            else:
                centralities=evolve_centrality_wrapper(graph, top=args.top_center)

        else:
            centralities=evolve_centrality_wrapper(graph, top=args.top_center)

        centralities_per_window.append(centralities)

        g_previous=graph
        centralities_prev=centralities


    if data_type is not None and dataset_name is not None:
        os.makedirs(f"data/{dataset_name}", exist_ok=True)
        with open(filename, "wb") as f:
            pickle.dump(centralities_per_window, f)
        print(f"Saved results to {filename}")


    return centralities_per_window

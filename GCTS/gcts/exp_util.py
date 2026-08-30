import torch
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from torch_geometric.utils import to_dense_adj, to_dense_batch
import torch.nn.functional as F




def plot_graph(adj_tensor, i, title="Graph", threshold=0.85):
    """
    Args:
        adj_tensor: A 2D tensor [N, N] or 3D tensor [1, N, N]
        i: identifier for filename
        title: title of the plot
        threshold: minimum value to consider an edge "existing"
    """
    if adj_tensor.dim() == 3:
        adj_tensor = adj_tensor[0]
        
    adj = adj_tensor.detach().cpu().numpy()
    
    if (adj > 1).any() or (adj < 0).any(): 
        adj = 1 / (1 + np.exp(-adj))

    adj[adj < threshold] = 0 

    G = nx.from_numpy_array(adj)
    
    if G.number_of_nodes() > 0:
        G.remove_nodes_from(list(nx.isolates(G)))

    plt.figure(figsize=(10, 8))
    plt.title(f"{title} (Threshold: {threshold}) - Index {i}")
    
    weights = [G[u][v]['weight'] * 2 for u, v in G.edges()]
    
    nx.draw(G, 
            with_labels=True, 
            node_color='skyblue', 
            edge_color='gray', 
            width=weights,
            node_size=500,
            font_size=10)
            
    plt.savefig(f"{title.lower()}_{i}.png")
    plt.close()

    return 



def build_dense_adjacency(G):
    A = to_dense_adj(G.edge_index, max_num_nodes=G.x.size(0))
    return A.squeeze(0)


def normalize_adjacency(A):
    I = torch.eye(A.size(0), device=A.device)
    A_hat = A + I
    
    D = torch.diag(torch.pow(A_hat.sum(1), -0.5))
    return D @ A_hat @ D


def kl_loss(mu, logvar):
    kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).mean()
    return kl

def reconstruction_loss(logits, A, pos_weight=None):
    return F.binary_cross_entropy_with_logits(
        logits, A, pos_weight=pos_weight
    )

def reconstruction_loss_v2(X_hat, X_target, mu, logvar):
    recon = F.mse_loss(X_hat, X_target)
    kl    = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return recon + kl



def total_loss(loss_bce, loss_kl,loss_temporal=None, beta=0, alpha=1.0):
    if loss_temporal is not None:
        return loss_bce + beta * loss_kl + alpha*loss_temporal
    else:
        return loss_bce + beta * loss_kl




def preprocess_graph(graph, i):
    A = to_dense_adj(graph.edge_index, batch=graph.batch, max_num_nodes=graph.x.size(1))

    X_dense, mask = to_dense_batch(graph.x, batch=graph.batch)
    I = torch.eye(A.size(-1), device=A.device).expand_as(A)
    A_hat = A + I


    D = A_hat.sum(dim=-1).clamp(min=1e-8) 
    D_inv_sqrt = D.pow(-0.5) 
    D_inv_sqrt[torch.isinf(D_inv_sqrt)] = 0.0

    D_mat = torch.diag_embed(D_inv_sqrt)
    A_norm = D_mat @ A_hat @ D_mat

    graph.x_batched = X_dense.transpose(1,2)
    graph.A_norm = A_norm
    graph.A=A
    return graph



def preprocess_graph_v2(graph):
    A = to_dense_adj(graph.edge_index, max_num_nodes=graph.x.size(1))
    A=A.squeeze(0)
    I = torch.eye(A.size(-1), device=A.device).expand_as(A)
    A_hat = A + I
    D = A_hat.sum(dim=-1).clamp(min=1e-8)
    D_inv_sqrt = D.pow(-0.5) 
    D_inv_sqrt[torch.isinf(D_inv_sqrt)] = 0.0
    D_mat = torch.diag_embed(D_inv_sqrt)
    A_norm = D_mat @ A_hat @ D_mat
    graph.A_norm = A_norm
    graph.A=A
    return graph


def get_pos_weight_score(graph):

    num_nodes = graph.x.shape[0]
    edge_list=graph.edge_index
    num_positive=len(edge_list[1])
    num_possible_edges = num_nodes * num_nodes
    num_negative = num_possible_edges - num_positive

    pos_weight = num_negative / (num_positive + 1e-8)
    return pos_weight

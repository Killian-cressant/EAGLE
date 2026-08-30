import networkx as nx
import torch
import numpy as np
import os
from pathlib import Path
from .cosi import build_cosi_matrix
import random


def fil_edges(Mat):
    edges_index=[]
    X1=[]
    X2=[]
    n=len(Mat)
    for k in range(len(Mat)):
        for j in range(len(Mat[0])):
            if k+j+1<n:
                if Mat[j][j+k+1]>0:
                    X1.append(j)
                    X2.append(j+k+1)
    edges_index.append(X1)
    edges_index.append(X2)
    edges_index=torch.tensor(edges_index)
    return edges_index




def get_edge_index(X: torch.Tensor , device: str= "cuda", name: str = None, mode: str = "train"):
    base_path = Path("data") / name / "edges"
    print(base_path)
    if base_path.exists() and base_path.is_dir():
        return None
    else:
        base_path.mkdir(parents=True, exist_ok=True)

        return None



def build_local_graph(X: torch.Tensor, device: str, top_k: int = 5) -> torch.Tensor:
    """
    Build edge_index using top-K absolute correlations.

    Args:
        X (torch.Tensor): [window_size, num_features]
        device (str): 'cpu' or 'cuda'
        k (int): number of neighbors per node

    Returns:
        edge_index (torch.Tensor): [2, num_edges]
    """
    X = X.to(device)

    n = X.shape[0]
    corr = (X.T @ X) / (n - 1)

    corr = torch.abs(corr)

    corr.fill_diagonal_(0)

    _, topk_indices = torch.topk(corr, k=top_k, dim=1)

    num_nodes = corr.size(0)

    row = torch.arange(num_nodes, device=device).unsqueeze(1).repeat(1, top_k)

    row = row.reshape(-1)
    col = topk_indices.reshape(-1)

    edge_index = torch.stack([row, col], dim=0)

    edge_index_rev = torch.stack([col, row], dim=0)
    edge_index = torch.cat([edge_index, edge_index_rev], dim=1)

    edge_index = torch.unique(edge_index, dim=1)
    return edge_index



def build_local_graph_v2(X: torch.Tensor, device: str, top_k: int = 5) -> torch.Tensor:
    """
    Build edge_index using top-K absolute correlations,
    but with a randomly chosen K (<= top_k).

    Args:
        X (torch.Tensor): [window_size, num_features]
        device (str): 'cpu' or 'cuda'
        top_k (int): maximum number of neighbors per node

    Returns:
        edge_index (torch.Tensor): [2, num_edges]
    """

    X = X.to(device)

    n = X.shape[0]
    corr = (X.T @ X) / (n - 1)
    corr = torch.abs(corr)
    corr.fill_diagonal_(0)

    k = random.randint(1, top_k)

    _, topk_indices = torch.topk(corr, k=k, dim=1)

    num_nodes = corr.size(0)

    row = torch.arange(num_nodes, device=device).unsqueeze(1).repeat(1, k)
    row = row.reshape(-1)
    col = topk_indices.reshape(-1)

    edge_index = torch.stack([row, col], dim=0)

    edge_index_rev = torch.stack([col, row], dim=0)
    edge_index = torch.cat([edge_index, edge_index_rev], dim=1)

    edge_index = torch.unique(edge_index, dim=1)

    return edge_index





def build_cosi_graph(X: torch.Tensor, w2v_path: str, device: str, dataset_name: str, cache_path: str = None, description_file: str = None, top_k: int = 200) -> torch.Tensor:
    cosi_graph = build_cosi_matrix(X=X.to(device),   
    w2v_path=w2v_path,
    device=device,
    cache_path = cache_path,
    dataset_name=dataset_name,
    description_file=description_file,
    top_k=top_k)

    return cosi_graph

import os
from typing import Tuple

import pandas as pd
import torch
import numpy as np
from gcts.config import CISCOPaths
from .data_processing import InterPolationMethods, downsample, preprocess_df



def rm_var_null(X: pd.DataFrame)-> pd.DataFrame:
    variances = X.var(axis=0)  

    X_filtered = X.loc[:, variances > 1e-6]

    if isinstance(X_filtered, pd.Series):
        X_filtered = X_filtered.to_frame()
    print(X_filtered.shape)
    return X_filtered

def remove_by_corr(df,cut_corr_param)-> pd.DataFrame:

  corr_mat=df.corr()
  upper = corr_mat.where(np.triu(np.ones(corr_mat.shape), k=1).astype(bool))

  to_drop = [column for column in upper.columns if any(upper[column] >= cut_corr_param)]
  
  df = df.drop(to_drop, axis=1)
  print(df.shape)
  return(df)


def keep_my_columns(df, cols):
    df=df[cols]
    return df



def load_cisco_df_train(
    name: str = CISCOPaths.name_train,
    path_to_dataset: str = CISCOPaths.base_path,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Loads the training dataset from the given path and returns a pandas DataFrame.
    Args:
        path_to_dataset: Path to the dataset files.
    Returns:
        A pandas DataFrame containing the training dataset.
    """
    file = os.path.join(path_to_dataset, name)
    df_train = pd.read_csv(file)
    df_train_labels =  np.zeros(len(df_train)) 
    df_train_labels = pd.DataFrame(df_train_labels, columns=["attack"]) 

    return df_train, df_train_labels


def load_cisco_df_val(
    name: str = CISCOPaths.name_val,
    path_to_dataset: str = CISCOPaths.base_path,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Loads the validation dataset from the given path and returns a pandas DataFrame.
    Args:
        path_to_dataset: Path to the dataset files.
    Returns:
        A pandas DataFrame containing the validation dataset.
    """
    file = os.path.join(path_to_dataset, name)
    df_val = pd.read_csv(file)
    df_val_labels = (df_val["attack"]).astype(int)
    df_val_labels = df_val_labels.to_frame()
    df_val = df_val.drop(columns=["attack"])
    return df_val, df_val_labels


def load_cisco_df_test(
    name: str = CISCOPaths.name_test,
    path_to_dataset: str = CISCOPaths.base_path,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    file = os.path.join(path_to_dataset, name)
    df_test = pd.read_csv(file)
    df_test_labels = (df_test["attack"]).astype(int)
    df_test_labels = df_test_labels.to_frame()
    df_test = df_test.drop(columns=["attack"])
    return df_test, df_test_labels


def load_cisco_df(
    path_to_dataset: str = CISCOPaths.base_path, val_size: float = 0.6
) -> Tuple[
    pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame
]:
    """
    Loads the dataset from the given path and returns a pandas DataFrame.
    Args:
        names: List of names of the files to load.
        path_to_dataset: Path to the dataset files.
    Returns:
        A pandas DataFrame containing the dataset.
    """
    df_train, df_train_labels = load_cisco_df_train(path_to_dataset=path_to_dataset)
    df_val, df_val_labels = load_cisco_df_val(path_to_dataset=path_to_dataset)
    df_test, df_test_labels = load_cisco_df_test(path_to_dataset=path_to_dataset)


    return df_train, df_train_labels, df_val, df_val_labels, df_test, df_test_labels


def load_cisco_training_data(
    path_to_dataset: str = CISCOPaths.base_path,
    normalize: bool = False,
    clean: bool = False,
    scaler=None,
    interpolate_method: InterPolationMethods | None = None,
    down_len: int | None = None,
    max_std: float | None = None,
    labels_widening: bool = False,
    cutoff_value: float | None = None,
) -> Tuple[torch.Tensor, ...]:
    """
    Load the data for the telco dataset, splitted into train, val and test.
    Args:
        base_path: The path where the datasets are stored.
        normalize: Whether to normalize the data. Default is False.
        clean: Whether to clean the data. Default is False.
        scaler: The scaler to use for normalization.
        interpolate_method: The method to use for interpolation.
        down_len: The length of the downsample window.
                If None, no downsampling is performed.
        max_std: Maximum standard deviation for data cleaning. Default is 0.0.
        labels_widening: Whether to widen the labels. Default is True.
        cutoff_value: The cutoff value for data cleaning. Default is 30.0.
    Returns:
        Tuple of training data, training labels, validation data, validation labels,
        and test data.
    """
    (
        df_train,
        df_train_labels,
        df_val,
        df_val_labels,
        df_test,
        df_test_labels,
    ) = load_cisco_df(path_to_dataset=path_to_dataset)

    columns_to_drop = ["time", "Unnamed: 0"]

    
    df_train = df_train.drop(columns=columns_to_drop)
    df_val = df_val.drop(columns=columns_to_drop)
    df_test = df_test.drop(columns=columns_to_drop)

    X_train, X_train_labels, scaler = preprocess_df(
        data_df=df_train,
        labels_df=df_train_labels,
        normalize=normalize,
        clean=clean,
        scaler=scaler,
        interpolate_method=interpolate_method,
        max_std=max_std,
        labels_widening=labels_widening,
        cutoff_value=cutoff_value,
    )
    X_val, X_val_labels, _ = preprocess_df(
        data_df=df_val,
        labels_df=df_val_labels,
        normalize=normalize,
        clean=clean,
        scaler=scaler,
        interpolate_method=interpolate_method,
        max_std=max_std,
        labels_widening=labels_widening,
        cutoff_value=cutoff_value,
    )
    X_test, X_test_labels, _ = preprocess_df(
        data_df=df_test,
        labels_df=df_test_labels,
        normalize=normalize,
        clean=False,
        scaler=scaler,
        interpolate_method=interpolate_method,
        max_std=max_std,
        labels_widening=labels_widening,
        cutoff_value=cutoff_value,
    )

    if X_train_labels is None or X_test_labels is None or X_val_labels is None:
        raise ValueError("CISCO labels are not being loaded.")


    if down_len is not None:
        if down_len < 1:
            raise ValueError("Downsample length must be greater than 0")

        print(f"Downsampling data by {down_len}")
        X_train, X_train_labels = downsample(X_train, X_train_labels, down_len)
        X_val, X_val_labels = downsample(X_val, X_val_labels, down_len)
        X_test, X_test_labels = downsample(X_test, X_test_labels, down_len)

    return X_train, X_val, X_test, X_train_labels, X_val_labels, X_test_labels


def build_cisco_edge_index(
    device: str, path: str = CISCOPaths.edge_index_path
) -> torch.Tensor:
    """
    Build the edge index based on the topological structure of Cisco system.

    Args:
        X: Input tensor containing the node features (not used in edge construction)
        device: Device to place the resulting edge_index tensor on ('cpu' or 'cuda')
        path: Path to save the edge index

    Returns:
        torch.Tensor: A tensor of shape [2, num_edges] containing the edge indices.
    """
    df, _ = load_cisco_df_val(path_to_dataset="../cisco")
    df = df.drop(columns=[" time"])
    node_names = df.columns.tolist()

    edges = []
    for i, name1 in enumerate(node_names):
        for j, name2 in enumerate(node_names):
            if i != j:
                stripped_name1 = name1.strip()
                stripped_name2 = name2.strip()

                if (
                    len(stripped_name1) == len(stripped_name2)
                    and stripped_name1[:-1] == stripped_name2[:-1]
                ):
                    edges.append([i, j])

    edge_index = torch.tensor(edges, dtype=torch.long).t()

    torch.save(edge_index.cpu(), path)

    return edge_index.to(device)

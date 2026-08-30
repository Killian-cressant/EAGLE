import os
from typing import Tuple

import pandas as pd
import torch

from gcts.config import WADIPaths
from .data_processing import InterPolationMethods, downsample, preprocess_df



#fixing fature 63 in wadi
def fix_feat63(feat):
    feat[feat > 1] = 1
    return feat


#fixing fature 106 in wadi
def fix_feat106(feat):
    for k in range(len(feat)):
        if abs(feat[k])>250:
            feat[k]=feat[k]-268
    return feat

"""take the first part for validation and the rest for test"""
def split_val_test(df, df_labels, val_size: float = 0.4):
    split_idx = int(len(df) * val_size)

    df_val = df.iloc[:split_idx]
    df_val_labels = df_labels.iloc[:split_idx]

    df_test = df.iloc[split_idx:]
    df_test_labels = df_labels.iloc[split_idx:]

    return df_val, df_val_labels, df_test, df_test_labels 


def load_wadi_df_train(
    name: str = WADIPaths.name_train,
    path_to_dataset: str = WADIPaths.base_path,
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
    df_train_labels = pd.DataFrame(0, index=df_train.index, columns=["Attack"])

    return df_train, df_train_labels

def load_wadi_df_val(
    name: str = WADIPaths.name_val,
    path_to_dataset: str = WADIPaths.base_path,
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

    df_val_labels = (df_val["Attack LABLE (1:No Attack, -1:Attack)"] == -1).astype(int)
    df_val_labels = df_val_labels.to_frame()
    df_val = df_val.drop(columns=["Attack LABLE (1:No Attack, -1:Attack)"])
    return df_val, df_val_labels



def load_wadi_df(
    path_to_dataset: str = WADIPaths.base_path, val_size: float = 0.6
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
    df_train, df_train_labels = load_wadi_df_train(path_to_dataset=path_to_dataset)
    df_val, df_val_labels = load_wadi_df_val(path_to_dataset=path_to_dataset)
    
    #wadi val and test are the same so we need to split them
    df_val, df_val_labels, df_test, df_test_labels=split_val_test(df_val, df_val_labels)

    return df_train, df_train_labels, df_val, df_val_labels, df_test, df_test_labels


def load_wadi_training_data(
    path_to_dataset: str = WADIPaths.base_path,
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
    Load the data for the wadi dataset, splitted into train, val and test.
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
    ) = load_wadi_df(path_to_dataset=path_to_dataset)

    df_train.columns = df_train.columns.str.strip()
    df_val.columns = df_val.columns.str.strip()
    df_test.columns = df_test.columns.str.strip()

    columns_to_drop = ["Time", "Date", "Row"]
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
        raise ValueError("WADI labels are not being loaded.")


    #fixing test dataset that has issue in 2 Time series:
    X_test[:, 63] = fix_feat63(X_test[:, 63])
    X_test[:, 106] = fix_feat106(X_test[:, 106])

    #X_test.drop

    if down_len is not None:
        if down_len < 1:
            raise ValueError("Downsample length must be greater than 0")

        print(f"Downsampling data by {down_len}")
        X_train, X_train_labels = downsample(X_train, X_train_labels, down_len)
        X_val, X_val_labels = downsample(X_val, X_val_labels, down_len)
        X_test, X_test_labels = downsample(X_test, X_test_labels, down_len)

    return X_train, X_val, X_test, X_train_labels, X_val_labels, X_test_labels




def build_wadi_edge_index(
    device: str, path: str = WADIPaths.edge_index_path
) -> torch.Tensor:
    """
    Build the edge index based on the topological structure of WADI system.

    Args:
        X: Input tensor containing the node features (not used in edge construction)
        device: Device to place the resulting edge_index tensor on ('cpu' or 'cuda')
        path: Path to save the edge index

    Returns:
        torch.Tensor: A tensor of shape [2, num_edges] containing the edge indices.
    """
    df, _ = load_wadi_df_val(path_to_dataset=WADIPaths.base_path)
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

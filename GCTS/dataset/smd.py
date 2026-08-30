import os
from typing import Tuple

import pandas as pd
import torch
import numpy as np
from gcts.config import SMDPaths
from .data_processing import InterPolationMethods, downsample, preprocess_df


def load_smd_df_train(
    name: str = SMDPaths.name_train,
    path_to_dataset: str = SMDPaths.base_path,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Loads the training dataset from the given path and returns a pandas DataFrame.
    Args:
        path_to_dataset: Path to the dataset files.
    Returns:
        A pandas DataFrame containing the training dataset.
    """
    file = os.path.join(path_to_dataset, name)
    df_np=np.load(file)
    df_train = pd.DataFrame(df_np)
    df_train_labels =  np.zeros(len(df_train)) 
    df_train_labels = pd.DataFrame(df_train_labels, columns=["attack"]) 

    return df_train, df_train_labels



def load_smd_df_test(
    name: str = SMDPaths.name_test,
    path_to_dataset: str = SMDPaths.base_path,
    labels: str = SMDPaths.name_test_label,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    file = os.path.join(path_to_dataset, name)
    df_np=np.load(file)
    df_test = pd.DataFrame(df_np)

    file_lab=os.path.join(path_to_dataset, labels)
    df_test_labels = np.load(file_lab)
    df_test_labels = pd.DataFrame(df_test_labels, columns=["attack"])

    return df_test, df_test_labels


def load_smd_df(
    path_to_dataset: str = SMDPaths.base_path, val_size: float = 0.3
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
    df_train, df_train_labels = load_smd_df_train(path_to_dataset=path_to_dataset)
    df_test, df_test_labels = load_smd_df_test(path_to_dataset=path_to_dataset)

    split_idx = int(len(df_test) * val_size)

    df_val = df_test.iloc[:split_idx].reset_index(drop=True)
    df_val_labels = df_test_labels.iloc[:split_idx].reset_index(drop=True)

    df_test_final = df_test.iloc[split_idx:].reset_index(drop=True)
    df_test_labels_final = df_test_labels.iloc[split_idx:].reset_index(drop=True)

    return df_train, df_train_labels, df_val, df_val_labels, df_test_final, df_test_labels_final


def load_smd_training_data(
    path_to_dataset: str = SMDPaths.base_path,
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
    Load the data for the SMD dataset, splitted into train, val and test.
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
    ) = load_smd_df(path_to_dataset=path_to_dataset)


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
        raise ValueError("SMD labels are not being loaded.")


    if down_len is not None:
        if down_len < 1:
            raise ValueError("Downsample length must be greater than 0")

        print(f"Downsampling data by {down_len}")
        X_train, X_train_labels = downsample(X_train, X_train_labels, down_len)
        X_val, X_val_labels = downsample(X_val, X_val_labels, down_len)
        X_test, X_test_labels = downsample(X_test, X_test_labels, down_len)

    return X_train, X_val, X_test, X_train_labels, X_val_labels, X_test_labels
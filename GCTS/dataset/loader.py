from gcts.types import *
from gcts.utils import get_logger
from gcts.config import get_dataset_config
import pandas as pd
from .swat import load_swat_training_data
from .telco import load_telco_training_data
from .cisco import load_cisco_training_data
from .wadi import load_wadi_training_data
from .smd import load_smd_training_data

logger = get_logger()


def load_training_data(
    dataset: Datasets,
    test_size: float = 0.1,
    val_size: float = 0.1,
    normalize: bool = False,
    clean: bool = False,
    interpolate_method: InterPolationMethods | None = None,
    down_len: int | None = None,
    max_std: float | None = None,
    labels_widening: bool = False,
    cutoff_value: float | None = None,
):
    """
    Load the training data for the given dataset.
    Args:
        dataset: The dataset to load.
        test_size: The size of the test set, from 0 to 1.
        val_size: The size of the validation set, from 0 to 1.
        normalize: Whether to normalize the data.
        clean: Whether to remove anomalies from the data.
        interpolate_method: The method to use to interpolate the missing values.
        down_len: The length of the downsample window.
                If None, no downsampling is performed.
        max_std: Maximum standard deviation for data cleaning
        labels_widening: Whether to widen the labels.
        cutoff_value: The cutoff value for data cleaning.
    Returns:
        The training, validation and test data as torch tensors.
    """


    if dataset == Datasets.TELCO:
        logger.warning(
            "Telco data is already splited, ignoring arguments: 'test_size'"
            ", 'val_size', 'shuffle' and 'random_state'"
        )
        return load_telco_training_data(normalize=normalize,
                                         clean=clean, interpolate_method=interpolate_method,
                                           down_len=down_len, max_std=max_std,
                                           labels_widening=labels_widening, cutoff_value=cutoff_value)
    elif dataset==Datasets.SWAT:
        return load_swat_training_data(normalize=normalize,
            clean=False,interpolate_method=interpolate_method,down_len=down_len,
            max_std=max_std,labels_widening=labels_widening,cutoff_value=cutoff_value)
    elif dataset==Datasets.CISCO:
        return load_cisco_training_data(normalize=False, clean= False, down_len=down_len, max_std=max_std,
                labels_widening=labels_widening, cutoff_value=cutoff_value)
    elif dataset==Datasets.WADI:
        return load_wadi_training_data(normalize=normalize,
            clean=False,interpolate_method=interpolate_method,down_len=down_len,
            max_std=max_std,labels_widening=labels_widening,cutoff_value=cutoff_value)
    elif dataset==Datasets.SMD:
        return load_smd_training_data(normalize=normalize,
            clean=False,interpolate_method=interpolate_method,down_len=down_len,
            max_std=max_std,labels_widening=labels_widening,cutoff_value=cutoff_value)
    else:
        raise ValueError(f"{dataset} is an unkown dataset")

        
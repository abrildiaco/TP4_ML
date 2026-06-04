import pandas as pd
import numpy as np

def standardize(dataset, train = True, xtr_mean = None, xtr_std = None):
    """
    Standardizes a dataset using standard scaling.
    When train=True, it learns the mean and standard deviation from the given dataset.
    When train=False, it uses the training statistics previously learned.

    Arguments:
        dataset (pd.DataFrame): dataset to standardize
        train (bool): whether the dataset is the training set
        x_train_mean (pd.Series | None): training mean for each column
        x_train_std (pd.Series | None): training standard deviation for each column

    Returns:
        pd.DataFrame | tuple[pd.DataFrame, pd.Series, pd.Series]: standardized dataset and, when train=True, training statistics
    """
    dataset = dataset.copy()

    # During training, compute the mean and std for each selected column
    if train:
        xtr_mean = dataset.mean()
        xtr_std = dataset.std()

        # If std = 0, replace it with 1 so the transformation becomes x - mean
        xtr_std_safe = xtr_std.replace(0, 1)

        xstd = (dataset - xtr_mean) / xtr_std_safe

        return xstd, xtr_mean, xtr_std_safe

    # During evaluation, use the statistics learned from train
    xtr_std_safe = xtr_std.replace(0, 1)
    standarized_dataset = (dataset - xtr_mean) / xtr_std_safe

    return standarized_dataset


def inverse_standardize(X_standardized, x_train_mean, x_train_std):
    """
    Converts standardized data back to the original pixel scale.
    If X_std = (X - mean_train) / std_train, then X = X_std * std_train + mean_train.

    Arguments:
        X_standardized (pd.DataFrame | np.ndarray): standardized data
        x_train_mean (pd.Series | np.ndarray): training mean for each original feature
        x_train_std (pd.Series | np.ndarray): training standard deviation for each original feature

    Returns:
        pd.DataFrame | np.ndarray: data reconstructed in the original pixel scale
    """
    # Convert the standardized data to NumPy to avoid pandas alignment issues
    X_values = X_standardized.to_numpy() if hasattr(X_standardized, "to_numpy") else np.asarray(X_standardized)

    # Convert the training statistics to NumPy
    mean_values = x_train_mean.to_numpy() if hasattr(x_train_mean, "to_numpy") else np.asarray(x_train_mean)
    std_values = x_train_std.to_numpy() if hasattr(x_train_std, "to_numpy") else np.asarray(x_train_std)

    # Apply the inverse transformation of standard scaling
    X_original_scale = X_values * std_values + mean_values

    # If the original mean came from a pandas Series, returns a DataFrame with the original column names.
    if hasattr(x_train_mean, "index"):
        return pd.DataFrame(X_original_scale, columns=x_train_mean.index)

    return X_original_scale
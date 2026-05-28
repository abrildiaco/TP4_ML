import pandas as pd

def standarize(dataset, train = True, xtr_mean = None, xtr_std = None):
    """
    standarized non-binary numerical columns using standard scaling.

    Arguments:
        dataset (pd.DataFrame): dataset containing the columns to standarize
        train (bool): whether the dataset is training data
        xtr_mean (pd.Series, optional): training set means for each selected column
        xtr_std (pd.Series, optional): training set standard deviations for each selected column

    Returns:
        dataset (pd.DataFrame): standarized dataset
        xtr_mean (pd.Series): [only if train=True]
        xtr_std_safe (pd.Series): [only if train=True]
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

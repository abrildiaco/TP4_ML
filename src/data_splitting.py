import numpy as np
import pandas as pd

def split_class_indices(class_indices, eval_size):
    """
    Splits indices from a single class into train and evaluation indices.

    Arguments:
        class_indices (np.ndarray): indices belonging to one class
        eval_size (float): proportion assigned to the evaluation set

    Returns:
        tuple[np.ndarray, np.ndarray]: train and evaluation indices for the class
    """
    # Compute how many observations from this class should go to evaluation
    n_eval = int(len(class_indices) * eval_size)

    # Take the first n_eval shuffled indices for evaluation and the rest for training
    eval_class_indices = class_indices[:n_eval]
    train_class_indices = class_indices[n_eval:]

    return train_class_indices, eval_class_indices


def stratified_train_eval_split(X, y, eval_size = 0.20, random_state = 42):
    """
    Splits the dataset into training and evaluation sets using a stratified split.
    The split preserves the same class proportions in both subsets.

    Arguments:
        X (pd.DataFrame): features
        y (pd.Series): labels
        eval_size (float): proportion of the dataset used for evaluation
        random_state (int): random seed for reproducibility

    Returns:
        tuple[tuple[pd.DataFrame, pd.Series], tuple[pd.DataFrame, pd.Series]]: train and evaluation splits
    """
    rng = np.random.default_rng(random_state)

    # Converts y to NumPy
    y_values = y.to_numpy() if hasattr(y, "to_numpy") else np.asarray(y)

    classes = np.unique(y_values)

    train_indices = []
    eval_indices = []

    for class_value in classes:
        # Find the row positions that belong to the current class
        class_indices = np.where(y_values == class_value)[0]

        # Shuffle the indices so the split is random inside each class
        rng.shuffle(class_indices)

        train_class_indices, eval_class_indices = split_class_indices(class_indices, eval_size)

        train_indices.extend(train_class_indices)
        eval_indices.extend(eval_class_indices)

    train_indices = np.array(train_indices)
    eval_indices = np.array(eval_indices)

    # Shuffle indeces to mix classes
    rng.shuffle(train_indices)
    rng.shuffle(eval_indices)

    # Use iloc to select rows based on the computed indices for both features and labels
    X_train = X.iloc[train_indices]
    X_eval = X.iloc[eval_indices]

    y_train = y.iloc[train_indices]
    y_eval = y.iloc[eval_indices]

    return (X_train, y_train), (X_eval, y_eval)
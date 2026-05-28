import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from src.graphics import CLASS_NAMES

def build_class_summary(X, y, class_names = CLASS_NAMES):
    """
    Builds a summary table with class counts, percentages and pixel intensity statistics.

    Arguments:
        X (pd.DataFrame | np.ndarray): flattened image data
        y (pd.Series | np.ndarray): image labels
        class_names (dict | None): mapping from numeric labels to class names

    Returns:
        pd.DataFrame: summary table by class
    """
    # Converts X and y to NumPy arrays, regardless of whether they come as pandas objects.
    X_values = X.to_numpy() if hasattr(X, "to_numpy") else np.asarray(X)
    y_values = y.to_numpy() if hasattr(y, "to_numpy") else np.asarray(y)

    # Counts how many observations belong to each class and sorts them by label.
    class_counts = pd.Series(y_values).value_counts().sort_index()

    class_summary = pd.DataFrame(index=class_counts.index)

    # Adds readable class names if a class dictionary is available.
    class_summary["class_name"] = [class_names.get(label, label) if class_names is not None else label for label in class_counts.index]

    class_summary["count"] = class_counts.values
    class_summary["percentage"] = (class_counts.values / len(y_values) * 100).round(2)

    mean_intensity = []
    std_intensity = []
    mean_nonzero_pixels = []

    for label in class_counts.index:
        # Selects all observations from the current class.
        X_class = X_values[y_values == label]

        # Computes the average pixel intensity for the current class.
        mean_intensity.append(X_class.mean())

        # Computes how much pixel intensity varies within the current class.
        std_intensity.append(X_class.std())

        # Counts non-black pixels per image and averages that count within the class.
        mean_nonzero_pixels.append((X_class > 0).sum(axis=1).mean())

    class_summary["mean_intensity"] = np.round(mean_intensity, 4)
    class_summary["std_intensity"] = np.round(std_intensity, 4)
    class_summary["mean_nonzero_pixels"] = np.round(mean_nonzero_pixels, 2)

    display(class_summary)
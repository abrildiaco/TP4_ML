import math
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

CLASS_NAMES = {
    0: "T-shirt/top",
    1: "Trouser",
    2: "Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandal",
    6: "Shirt",
    7: "Sneaker",
    8: "Bag",
    9: "Ankle boot"
}

def plot_images(X, y=None, n_images = 15, indices = None, random_state = 42, image_shape = (28, 28),
                class_names = CLASS_NAMES, n_cols = 5, cmap = "gray", title = None):
    """
    Plots an arbitrary number of images from a dataset of flattened images.
    If indices are not provided, it selects random images without replacement.

    Arguments:
        X (pd.DataFrame | np.ndarray): flattened image data
        y (pd.Series | np.ndarray | None): image labels
        n_images (int): number of images to plot when indices are not provided
        indices (list | np.ndarray | None): specific image indices to plot
        random_state (int): random seed for reproducibility
        image_shape (tuple[int, int]): original image shape
        class_names (dict | None): mapping from numeric labels to class names
        n_cols (int): number of columns in the figure
        cmap (str): matplotlib colormap
        title (str | None): general title for the full figure

    Returns:
        None
    """
    # Converts X to a NumPy array if it is a pandas DataFrame.
    X_values = X.to_numpy() if hasattr(X, "to_numpy") else np.asarray(X)

    # Converts y to a NumPy array if labels were provided.
    if y is not None:
        y_values = y.to_numpy() if hasattr(y, "to_numpy") else np.asarray(y)

    # If no specific indices are provided, randomly selects n_images observations.
    if indices is None:
        rng = np.random.default_rng(random_state)

        n_images = min(n_images, len(X_values))
        indices = rng.choice(len(X_values), size=n_images, replace=False)

    else:
        indices = np.asarray(indices)
        n_images = len(indices)

    n_rows = math.ceil(n_images / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize = (2.2 * n_cols, 2.4 * n_rows + 1))

    if title is not None:
        fig.suptitle(title, fontsize = 14, fontweight = "bold")

    # Flattens axes so it can be iterated as a one-dimensional list.
    axes = np.array(axes).reshape(-1)

    # Plots each selected image in one subplot.
    for ax, idx in zip(axes, indices):
        # Converts the flattened 784-pixel vector back into a 28x28 image.
        image = X_values[idx].reshape(image_shape)

        ax.imshow(image, cmap=cmap)
        ax.axis("off")

        if y is not None:
            label = y_values[idx]

            label_name = class_names.get(label, label) if class_names is not None else label

            ax.set_title(f"{label}: {label_name}", fontsize = 9)

    # Turns off empty subplots when the grid has more spaces than images.
    for ax in axes[n_images:]:
        ax.axis("off")

    if title is not None:
        plt.tight_layout(rect = (0, 0, 1, 0.95))
    else:
        plt.tight_layout()

    plt.show()


def plot_images_by_class(X, y, classes = None, n_per_class = 5, random_state = 42, image_shape = (28, 28),
                         class_names = CLASS_NAMES, cmap = "gray", title = None):
    """
    Plots random image samples grouped by class.
    Each row corresponds to one class and each column to one sampled image.

    Arguments:
        X (pd.DataFrame | np.ndarray): flattened image data
        y (pd.Series | np.ndarray): image labels
        classes (list | np.ndarray | None): classes to plot
        n_per_class (int): number of images to plot per class
        random_state (int): random seed for reproducibility
        image_shape (tuple[int, int]): original image shape
        class_names (dict | None): mapping from numeric labels to class names
        cmap (str): matplotlib colormap
        title (str | None): general title for the full figure

    Returns:
        None
    """
    # Converts X and y to NumPy arrays, regardless of whether they come as pandas objects.
    X_values = X.to_numpy() if hasattr(X, "to_numpy") else np.asarray(X)
    y_values = y.to_numpy() if hasattr(y, "to_numpy") else np.asarray(y)

    # If no classes are provided, selects the first 5 classes by label.
    if classes is None:
        classes = np.unique(y_values)[:5]

    # Uses a random generator to make the sampled images reproducible.
    rng = np.random.default_rng(random_state)

    n_rows = len(classes)
    n_cols = n_per_class + 1

    # Creates a narrow first column for class labels and regular columns for images.
    width_ratios = [0.55] + [1] * n_per_class
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(1.5 * n_cols, 2 * n_rows), gridspec_kw={"width_ratios": width_ratios})

    if title is not None:
        fig.suptitle(title, fontsize=14, fontweight="bold")

    # Ensures axes is always a 2D array, even if there is only one class.
    axes = np.asarray(axes).reshape(n_rows, n_cols)

    for row, class_label in enumerate(classes):
        # Builds the row label using the numeric label and the class name.
        class_name = class_names.get(class_label, class_label) if class_names is not None else class_label
        row_label = f"{class_label}: {class_name}"

        # Uses the first column only to display the class label.
        label_ax = axes[row, 0]
        label_ax.axis("off")
        label_ax.text(1, 0.5, row_label, fontsize=10, fontweight="bold", ha="right", va="center", transform=label_ax.transAxes)

        # Finds all dataset indices that belong to the current class.
        class_indices = np.where(y_values == class_label)[0]

        # Avoids asking for more images than the selected class contains.
        sample_size = min(n_per_class, len(class_indices))

        # Samples image indices from the current class without replacement.
        selected_indices = rng.choice(class_indices, size=sample_size, replace=False)

        for col in range(n_per_class):
            ax = axes[row, col + 1]
            ax.axis("off")

            # If there are fewer images than n_per_class, leaves the remaining axes empty.
            if col >= sample_size:
                continue

            idx = selected_indices[col]

            # Converts the flattened 784-pixel vector back into a 28x28 image.
            image = X_values[idx].reshape(image_shape)

            ax.imshow(image, cmap=cmap)

    fig.subplots_adjust(left=0.03, right=0.99, top=0.93, bottom=0.02, wspace=0.08, hspace=0.08)

    plt.show()


def plot_class_distribution(y, class_names=CLASS_NAMES, title="Class distribution"):
    """
    Plots the number of samples per class.

    Arguments:
        y (pd.Series | np.ndarray): image labels
        class_names (dict | None): mapping from numeric labels to class names
        title (str): title for the figure

    Returns:
        pd.Series: number of samples per class
    """
    # Converts y to a pandas Series to use value_counts and keep the code simple.
    y_series = pd.Series(y)

    # Counts how many observations belong to each class and sorts them by label.
    class_counts = y_series.value_counts().sort_index()

    # Creates readable labels using the class name when available.
    labels = [f"{label}: {class_names.get(label, label)}" if class_names is not None else str(label) for label in class_counts.index]

    plt.figure(figsize=(10, 4))
    bars = plt.bar(labels, class_counts.values, color = "#05216d")

    # Adds the count above each bar to make the distribution easier to read.
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, height, int(height), ha="center", va="bottom", fontsize=9)

    plt.title(title, fontsize=14, fontweight="bold")
    plt.xlabel("Class")
    plt.ylabel("Number of samples")
    plt.xticks(rotation=45, ha="right")

    # Adds a bit of vertical space so the text above the bars does not touch the border.
    plt.ylim(0, class_counts.max() * 1.1)

    plt.tight_layout()
    plt.show()

    return class_counts


def plot_mean_image_by_class(X, y, classes=None, image_shape=(28, 28), class_names=CLASS_NAMES, 
                             cmap="gray", title="Imagen promedio por clase"):
    """
    Plots the average image for each selected class.

    Arguments:
        X (pd.DataFrame | np.ndarray): flattened image data
        y (pd.Series | np.ndarray): image labels
        classes (list | np.ndarray | None): classes to plot
        image_shape (tuple[int, int]): original image shape
        class_names (dict | None): mapping from numeric labels to class names
        cmap (str): matplotlib colormap
        title (str): title for the figure

    Returns:
        None
    """
    # Converts X and y to NumPy arrays, regardless of whether they come as pandas objects.
    X_values = X.to_numpy() if hasattr(X, "to_numpy") else np.asarray(X)
    y_values = y.to_numpy() if hasattr(y, "to_numpy") else np.asarray(y)

    # If no classes are provided, plots all classes sorted by label.
    if classes is None:
        classes = np.unique(y_values)

    n_cols = 5
    n_rows = math.ceil(len(classes) / n_cols)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.2 * n_cols, 2.55 * n_rows))
    fig.suptitle(title, fontsize=14, fontweight="bold")

    # Flattens axes so it can be iterated as a one-dimensional list.
    axes = np.asarray(axes).reshape(-1)

    for ax, label in zip(axes, classes):
        # Selects all images from the current class.
        X_class = X_values[y_values == label]

        # Computes the average pixel value across all images in the class.
        mean_image = X_class.mean(axis=0).reshape(image_shape)

        class_name = class_names.get(label, label) if class_names is not None else label

        ax.imshow(mean_image, cmap=cmap)
        ax.axis("off")

        # Places the class label below the image to avoid overlap between rows.
        ax.text(0.5, -0.08, f"{label}: {class_name}", fontsize=9, ha="center", va="top", transform=ax.transAxes)

    # Turns off empty subplots when the grid has more spaces than classes.
    for ax in axes[len(classes):]:
        ax.axis("off")

    # Controls spacing manually so labels and title do not overlap.
    fig.subplots_adjust(left=0.02, right=0.98, top=0.86, bottom=0.08, wspace=0.18, hspace=0.38)

    plt.show()
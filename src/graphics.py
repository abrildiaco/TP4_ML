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

def plot_images(X, y=None, n_images=15, indices=None, random_state=42, image_shape=(28, 28),
                class_names=CLASS_NAMES, n_cols=5, cmap="gray", title=None, compare_with=None, comparison_labels=("Original", "Reconstruccion"), clip_compare=True):
    """
    Plots an arbitrary number of images from a dataset of flattened images.
    If compare_with is provided, it compares the original images with one or more reconstructed versions.

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
        compare_with (pd.DataFrame | np.ndarray | list | dict | None): reconstructed images to compare against X
        comparison_labels (tuple | list): row labels used when compare_with is provided
        clip_compare (bool): whether to clip reconstructed images to the range [0, 1]

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

    # Keeps the original behavior when no comparison images are provided.
    if compare_with is None:
        n_rows = math.ceil(n_images / n_cols)
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.2 * n_cols, 2.4 * n_rows + 1))

        if title is not None:
            fig.suptitle(title, fontsize=14, fontweight="bold")

        axes = np.array(axes).reshape(-1)

        for ax, idx in zip(axes, indices):
            image = X_values[idx].reshape(image_shape)

            ax.imshow(image, cmap=cmap)
            ax.axis("off")

            if y is not None:
                label = y_values[idx]
                label_name = class_names.get(label, label) if class_names is not None else label
                ax.set_title(f"{label}: {label_name}", fontsize=9)

        for ax in axes[n_images:]:
            ax.axis("off")

        if title is not None:
            plt.tight_layout(rect=(0, 0, 1, 0.95))
        else:
            plt.tight_layout()

        plt.show()
        return

    # Builds the list of datasets and row labels to compare.
    plot_data = [X_values]
    row_labels = [comparison_labels[0]]

    if isinstance(compare_with, dict):
        for label, data in compare_with.items():
            data_values = data.to_numpy() if hasattr(data, "to_numpy") else np.asarray(data)
            plot_data.append(data_values)
            row_labels.append(label)

    elif isinstance(compare_with, (list, tuple)) and not isinstance(compare_with, np.ndarray):
        for i, data in enumerate(compare_with):
            data_values = data.to_numpy() if hasattr(data, "to_numpy") else np.asarray(data)
            label = comparison_labels[i + 1] if i + 1 < len(comparison_labels) else f"Comparación {i + 1}"
            plot_data.append(data_values)
            row_labels.append(label)

    else:
        data_values = compare_with.to_numpy() if hasattr(compare_with, "to_numpy") else np.asarray(compare_with)
        label = comparison_labels[1] if len(comparison_labels) > 1 else "Comparación"
        plot_data.append(data_values)
        row_labels.append(label)

    n_comparison_rows = len(plot_data)
    n_grid_rows = math.ceil(n_images / n_cols)

    fig, axes = plt.subplots(n_comparison_rows * n_grid_rows, n_cols, figsize=(2.2 * n_cols, 2.15 * n_comparison_rows * n_grid_rows + 1))

    if title is not None:
        fig.suptitle(title, fontsize=14, fontweight="bold")

    axes = np.asarray(axes).reshape(n_comparison_rows * n_grid_rows, n_cols)

    for plot_position, idx in enumerate(indices):
        grid_row = plot_position // n_cols
        grid_col = plot_position % n_cols

        for comparison_row in range(n_comparison_rows):
            ax = axes[grid_row * n_comparison_rows + comparison_row, grid_col]

            image = plot_data[comparison_row][idx].reshape(image_shape)

            # Reconstructions can slightly exceed the valid pixel range.
            if comparison_row > 0 and clip_compare:
                image = np.clip(image, 0, 1)

            ax.imshow(image, cmap=cmap)
            ax.axis("off")

            # Adds class labels only over the original images.
            if comparison_row == 0 and y is not None:
                label = y_values[idx]
                label_name = class_names.get(label, label) if class_names is not None else label
                ax.set_title(f"{label}: {label_name}", fontsize=9)

            # Adds row labels only at the left of each block.
            if grid_col == 0:
                ax.text(-0.25, 0.5, row_labels[comparison_row], fontsize=10, fontweight="bold", ha="right", va="center", transform=ax.transAxes)

    # Turns off empty subplots in the last block.
    for empty_position in range(n_images, n_grid_rows * n_cols):
        grid_row = empty_position // n_cols
        grid_col = empty_position % n_cols

        for comparison_row in range(n_comparison_rows):
            axes[grid_row * n_comparison_rows + comparison_row, grid_col].axis("off")

    if title is not None:
        fig.subplots_adjust(left=0.08, right=0.99, top=0.92, bottom=0.03, wspace=0.08, hspace=0.15)
    else:
        fig.subplots_adjust(left=0.08, right=0.99, top=0.98, bottom=0.03, wspace=0.08, hspace=0.15)

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


def plot_cumulative_variance(pca_params, variance_threshold = 0.95, title = "Varianza explicada acumulada vs Componentes"):
    """
    Plots the cumulative explained variance as a function of the number of principal components.
    It also marks the selected variance threshold and the number of components needed to reach it.

    Arguments:
        pca_params (dict): fitted PCA parameters returned by fit_pca
        variance_threshold (float): cumulative explained variance threshold
        title (str): title for the figure

    Returns:
        None
    """
    # Gets the cumulative explained variance from the fitted PCA parameters.
    cumulative_variance = pca_params["cumulative_variance_ratio"]

    # Finds the first component that reaches the selected variance threshold.
    m_components = np.searchsorted(cumulative_variance, variance_threshold) + 1

    plt.figure(figsize=(8, 4))

    # Plots cumulative variance against the number of components.
    plt.plot(np.arange(1, len(cumulative_variance) + 1), cumulative_variance, linewidth=2)

    # Marks the variance threshold and the selected number of components.
    plt.axhline(variance_threshold, color="red", linestyle="--", label=f"{int(variance_threshold * 100)}% de varianza")
    plt.axvline(m_components, color="gray", linestyle="--", label=f"{m_components} componentes")

    plt.title(title, fontsize=14)
    plt.xlabel("Número de componentes")
    plt.ylabel("Varianza explicada acumulada")
    plt.ylim(0, 1.02)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_k_means_inertia_analysis(kmeans_pca_results, kmeans_ae_results, title="Análisis de inercia de k-Means"):
    """
    Plots raw and normalized k-means inertia for PCA and AE latent representations in a single figure.

    Arguments:
        kmeans_pca_results (dict): k-means results over PCA latent data indexed by K
        kmeans_ae_results (dict): k-means results over AE latent data indexed by K
        title (str): general title for the figure

    Returns:
        None
    """
    k_values = sorted(kmeans_pca_results.keys())

    # Extract inertia values ordered by K
    pca_inertia = np.array([kmeans_pca_results[k]["inertia"] for k in k_values])
    ae_inertia = np.array([kmeans_ae_results[k]["inertia"] for k in k_values])

    # Normalize each curve by its first value
    pca_inertia_norm = pca_inertia / pca_inertia[0]
    ae_inertia_norm = ae_inertia / ae_inertia[0]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
    fig.suptitle(title, fontsize=15, fontweight="bold")

    # Raw inertia
    axes[0].plot(k_values, pca_inertia, marker="o", label="PCA")
    axes[0].plot(k_values, ae_inertia, marker="o", label="AE")
    axes[0].set_title("Inercia según K", fontweight="bold")
    axes[0].set_xlabel("K")
    axes[0].set_ylabel("Inertia")
    axes[0].set_xticks(k_values)
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    # Normalized inertia
    axes[1].plot(k_values, pca_inertia_norm, marker="o", label="PCA")
    axes[1].plot(k_values, ae_inertia_norm, marker="o", label="AE")
    axes[1].set_title("Inercia normalizada según K", fontweight="bold")
    axes[1].set_xlabel("K")
    axes[1].set_ylabel("Normalized inertia")
    axes[1].set_xticks(k_values)
    axes[1].grid(alpha=0.3)
    axes[1].legend()

    plt.tight_layout(rect=(0, 0, 1, 0.92))
    plt.show()


def plot_gmm_log_likelihood_comparison(pca_results, ae_results, title="Log-likelihood de GMM según K"):
    """
    Plots GMM log-likelihood for PCA and AE latent representations.

    Arguments:
        pca_results (dict): GMM results over PCA latent data indexed by k
        ae_results (dict): GMM results over AE latent data indexed by k
        title (str): title for the figure

    Returns:
        None
    """
    k_values = sorted(pca_results.keys())

    # Extracts log-likelihood values ordered by k
    pca_log_likelihood = [pca_results[k]["log_likelihood"] for k in k_values]
    ae_log_likelihood = [ae_results[k]["log_likelihood"] for k in k_values]

    plt.figure(figsize=(8, 4))

    plt.plot(k_values, pca_log_likelihood, marker="o", label="PCA")
    plt.plot(k_values, ae_log_likelihood, marker="o", label="AE")

    plt.title(title, fontsize=14, fontweight="bold")
    plt.xlabel("K")
    plt.ylabel("Log-likelihood")
    plt.xticks(k_values)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_clustering_k_analysis(kmeans_pca_gain, kmeans_ae_gain, gmm_pca_gain, gmm_ae_gain, kmeans_pca_silhouette, kmeans_ae_silhouette, gmm_pca_silhouette, gmm_ae_silhouette, title="Análisis de K para clustering"):
    """
    Plots marginal gain and Silhouette score curves for k-Means and GMM in a single figure.

    Arguments:
        kmeans_pca_gain (pd.DataFrame): marginal gain table for k-Means with PCA
        kmeans_ae_gain (pd.DataFrame): marginal gain table for k-Means with AE
        gmm_pca_gain (pd.DataFrame): marginal gain table for GMM with PCA
        gmm_ae_gain (pd.DataFrame): marginal gain table for GMM with AE
        kmeans_pca_silhouette (pd.DataFrame): Silhouette score table for k-Means with PCA
        kmeans_ae_silhouette (pd.DataFrame): Silhouette score table for k-Means with AE
        gmm_pca_silhouette (pd.DataFrame): Silhouette score table for GMM with PCA
        gmm_ae_silhouette (pd.DataFrame): Silhouette score table for GMM with AE
        title (str): general title for the figure

    Returns:
        None
    """
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))

    fig.suptitle(title, fontsize=15, fontweight="bold")

    # k-Means marginal gain
    axes[0, 0].plot(kmeans_pca_gain["K"], kmeans_pca_gain["marginal_gain"], marker="o", label="PCA")
    axes[0, 0].plot(kmeans_ae_gain["K"], kmeans_ae_gain["marginal_gain"], marker="o", label="AE")
    axes[0, 0].set_title("Ganancia marginal de k-Means (Inercia)", fontweight="bold")
    axes[0, 0].set_xlabel("K")
    axes[0, 0].set_ylabel("Marginal gain")
    axes[0, 0].grid(alpha=0.3)
    axes[0, 0].legend()

    # GMM marginal gain
    axes[0, 1].plot(gmm_pca_gain["K"], gmm_pca_gain["marginal_gain"], marker="o", label="PCA")
    axes[0, 1].plot(gmm_ae_gain["K"], gmm_ae_gain["marginal_gain"], marker="o", label="AE")
    axes[0, 1].set_title("Ganancia marginal de GMM (Log-Likelihood)", fontweight="bold")
    axes[0, 1].set_xlabel("K")
    axes[0, 1].set_ylabel("Marginal gain")
    axes[0, 1].grid(alpha=0.3)
    axes[0, 1].legend()

    # k-Means Silhouette score
    axes[1, 0].plot(kmeans_pca_silhouette["K"], kmeans_pca_silhouette["silhouette_score"], marker="o", label="PCA", color = "red")
    axes[1, 0].plot(kmeans_ae_silhouette["K"], kmeans_ae_silhouette["silhouette_score"], marker="o", label="AE", color = "green")
    axes[1, 0].set_title("Silhouette score de k-Means", fontweight="bold")
    axes[1, 0].set_xlabel("K")
    axes[1, 0].set_ylabel("Silhouette score")
    axes[1, 0].grid(alpha=0.3)
    axes[1, 0].legend()

    # GMM Silhouette score
    axes[1, 1].plot(gmm_pca_silhouette["K"], gmm_pca_silhouette["silhouette_score"], marker="o", label="PCA", color = "red")
    axes[1, 1].plot(gmm_ae_silhouette["K"], gmm_ae_silhouette["silhouette_score"], marker="o", label="AE", color = "green")
    axes[1, 1].set_title("Silhouette score de GMM", fontweight="bold")
    axes[1, 1].set_xlabel("K")
    axes[1, 1].set_ylabel("Silhouette score")
    axes[1, 1].grid(alpha=0.3)
    axes[1, 1].legend()

    for ax in axes.ravel():
        ax.set_xticks(sorted(kmeans_pca_silhouette["K"].unique()))

    plt.tight_layout(rect=(0, 0, 1, 0.95))
    plt.show()


def plot_tsne_cluster_comparison(Y, labels_1, labels_2, title_1="k-Means", title_2="GMM", general_title="t-SNE sobre PCA latente"):
    """
    Plots the same 2D t-SNE embedding colored by two different clustering assignments.

    Arguments:
        Y (np.ndarray): 2D t-SNE coordinates
        labels_1 (np.ndarray): first clustering labels
        labels_2 (np.ndarray): second clustering labels
        title_1 (str): title for the first subplot
        title_2 (str): title for the second subplot
        general_title (str): general title for the figure

    Returns:
        None
    """
    labels_1 = np.asarray(labels_1)
    labels_2 = np.asarray(labels_2)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))

    fig.suptitle(general_title, fontsize=15, fontweight="bold")

    scatter_1 = axes[0].scatter(Y[:, 0], Y[:, 1], c=labels_1, cmap="tab10", s=12, alpha=0.75)
    axes[0].set_title(title_1, fontweight="bold")
    axes[0].set_xlabel("t-SNE 1")
    axes[0].set_ylabel("t-SNE 2")
    axes[0].grid(alpha=0.2)
    cbar_1 = fig.colorbar(scatter_1, ax=axes[0])
    cbar_1.set_label("Cluster")

    scatter_2 = axes[1].scatter(Y[:, 0], Y[:, 1], c=labels_2, cmap="tab10", s=12, alpha=0.75)
    axes[1].set_title(title_2, fontweight="bold")
    axes[1].set_xlabel("t-SNE 1")
    axes[1].set_ylabel("t-SNE 2")
    axes[1].grid(alpha=0.2)
    cbar_2 = fig.colorbar(scatter_2, ax=axes[1])
    cbar_2.set_label("Cluster")

    plt.tight_layout(rect=(0, 0, 1, 0.92))
    plt.show()


def plot_cluster_size_comparison(kmeans_size_table, gmm_size_table, title="Tamaño de clusters"):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
    fig.suptitle(title, fontsize=15, fontweight="bold")

    for ax, table, subtitle in zip(axes, [kmeans_size_table, gmm_size_table], ["k-Means", "GMM"]):
        bars = ax.bar(table["cluster"], table["count"], color="#2C7FB8")

        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, height + 8, f"{int(height)}", ha="center", va="bottom", fontsize=9)

        ax.set_title(subtitle, fontweight="bold")
        ax.set_xlabel("Cluster")
        ax.set_ylabel("Cantidad de muestras")
        ax.set_xticks(table["cluster"])
        ax.grid(axis="y", alpha=0.3)

    plt.tight_layout(rect=(0, 0, 1, 0.92))
    plt.show()


def plot_cluster_class_heatmap_comparison(kmeans_class_table, gmm_class_table, title="Composición de clases por cluster"):
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.8), constrained_layout=True)
    fig.suptitle(title, fontsize=15, fontweight="bold")

    vmax = max(kmeans_class_table.values.max(), gmm_class_table.values.max())
    tables = [kmeans_class_table, gmm_class_table]
    subtitles = ["k-Means", "GMM"]

    for ax, table, subtitle in zip(axes, tables, subtitles):
        image = ax.imshow(table.values, aspect="auto", cmap="Blues", vmin=0, vmax=vmax)

        ax.set_title(subtitle, fontweight="bold")
        ax.set_xlabel("Clase real")
        ax.set_ylabel("Cluster")
        ax.set_xticks(np.arange(table.shape[1]))
        ax.set_xticklabels(table.columns, rotation=45, ha="right", fontsize=9)
        ax.set_yticks(np.arange(table.shape[0]))
        ax.set_yticklabels(table.index)

        for i in range(table.shape[0]):
            for j in range(table.shape[1]):
                value = table.values[i, j]
                if value > 0:
                    ax.text(j, i, int(value), ha="center", va="center", fontsize=8, color="black")

    cbar = fig.colorbar(image, ax=axes, shrink=0.85, pad=0.02)
    cbar.set_label("Cantidad")

    plt.show()


def plot_cluster_purity_comparison(kmeans_purity_table, gmm_purity_table, title="Pureza de clusters"):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
    fig.suptitle(title, fontsize=15, fontweight="bold")

    for ax, table, subtitle in zip(axes, [kmeans_purity_table, gmm_purity_table], ["k-Means", "GMM"]):
        bars = ax.bar(table["cluster"], table["purity"], color="#2C7FB8")

        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, height + 0.015, f"{height:.2f}", ha="center", va="bottom", fontsize=9)

        ax.set_title(subtitle, fontweight="bold")
        ax.set_xlabel("Cluster")
        ax.set_ylabel("Pureza")
        ax.set_ylim(0, 1.08)
        ax.set_xticks(table["cluster"])
        ax.grid(axis="y", alpha=0.3)

    plt.tight_layout(rect=(0, 0, 1, 0.92))
    plt.show()


def plot_images_by_cluster(X_images, cluster_labels, y_true=None, cluster_id=0, n_images=10, random_state=42, title=None):
    """
    Plots random original images assigned to one selected cluster.

    Arguments:
        X_images (pd.DataFrame | np.ndarray): original image data
        cluster_labels (np.ndarray): cluster labels
        y_true (pd.Series | np.ndarray | None): true labels
        cluster_id (int): cluster to visualize
        n_images (int): number of images to plot
        random_state (int): random seed for reproducibility
        title (str | None): title for the figure

    Returns:
        None
    """
    labels = np.asarray(cluster_labels)

    # Gets positions assigned to the selected cluster
    cluster_indices = np.where(labels == cluster_id)[0]

    if len(cluster_indices) == 0:
        print(f"El cluster {cluster_id} no tiene muestras")
        return

    rng = np.random.default_rng(random_state)
    sample_size = min(n_images, len(cluster_indices))

    # Samples images from the selected cluster without replacement
    selected_indices = rng.choice(cluster_indices, size=sample_size, replace=False)

    if title is None:
        title = f"Muestras del cluster {cluster_id}"

    plot_images(X_images, y_true, n_images=sample_size, indices=selected_indices, n_cols=min(5, sample_size), title=title)
import numpy as np
import pandas as pd
from src.graphics import CLASS_NAMES
from src.clustering import k_means, gmm, compute_silhouette_score
from IPython.display import display

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


def run_k_means_for_k_range(X, k_values, max_iters = 200, tol = 1e-4, init = "kmeans++"):
    """
    Runs k-means for several values of k and stores the results.

    Arguments:
        X (pd.DataFrame | np.ndarray): data matrix
        k_values (list | range | np.ndarray): values of k to evaluate
        max_iters (int): maximum number of iterations
        tol (float): convergence tolerance for centroid movement
        init (str): centroid initialization method, either "random" or "kmeans++"

    Returns:
        dict: k-means results indexed by k
    """
    results_by_k = {}

    for k in k_values:
        # Run k-means for the current value of k
        results = k_means(X, k, max_iters = max_iters, tol = tol, init = init)

        # Stores the result using k as key
        results_by_k[k] = results

        print(f"K={k} | inertia={results['inertia']:.2f} | iterations={results['n_iters']}")

    return results_by_k


def display_k_means_comparison(pca_results, ae_results, title = "Resultados de k-Means"):
    """
    Builds and displays a comparison table for k-means results obtained with PCA and AE representations.

    Arguments:
        pca_results (dict): k-means results over PCA latent data indexed by k
        ae_results (dict): k-means results over AE latent data indexed by k
        title (str): title displayed above the table

    Returns:
        pd.DataFrame: comparison table with inertia and iterations for PCA and AE
    """
    rows = []

    for k in pca_results.keys():
        # Stores PCA and AE results for the same value of k
        rows.append({
            "K": k,
            "PCA inertia": pca_results[k]["inertia"],
            "PCA iterations": pca_results[k]["n_iters"],
            "AE inertia": ae_results[k]["inertia"],
            "AE iterations": ae_results[k]["n_iters"]
        })

    comparison = pd.DataFrame(rows)

    # Sorts the table by k to make the comparison easier to read
    comparison = comparison.sort_values("K").reset_index(drop=True)

    print(title)

    display(
        comparison.style
        .hide(axis="index")
        .format({
            "K": "{:.0f}",
            "PCA inertia": "{:.3f}",
            "PCA iterations": "{:.0f}",
            "AE inertia": "{:.3f}",
            "AE iterations": "{:.0f}"
        })
    )


def run_gmm_for_k_range(X, k_values, max_iters = 200, tol=1e-10, random_state = 42, epsilon = 1e-12):
    """
    Runs GMM for several values of k and stores the results.

    Arguments:
        X (pd.DataFrame | np.ndarray): data matrix
        k_values (list | range | np.ndarray): values of k to evaluate
        max_iters (int): maximum number of EM iterations
        tol (float): convergence tolerance for log-likelihood improvement
        random_state (int): random seed for reproducibility
        epsilon (float): small value used for numerical stability

    Returns:
        dict: GMM results indexed by k
    """
    results_by_k = {}

    for k in k_values:
        # Runs GMM for the current value of k
        results = gmm(X, k, max_iters = max_iters, tol = tol, random_state = random_state, epsilon = epsilon)

        # Stores the result using k as key
        results_by_k[k] = results

        print(f"K={k} | log_likelihood={results['log_likelihood']:.2f} | iterations={results['n_iters']}")

    return results_by_k


def display_gmm_comparison(pca_results, ae_results, title = "Resultados de GMM"):
    """
    Builds and displays a comparison table for GMM results obtained with PCA and AE representations.

    Arguments:
        pca_results (dict): GMM results over PCA latent data indexed by k
        ae_results (dict): GMM results over AE latent data indexed by k
        title (str): title displayed above the table

    Returns:
        pd.DataFrame: comparison table with log-likelihood and iterations for PCA and AE
    """
    rows = []

    for k in pca_results.keys():
        # Stores PCA and AE results for the same value of k
        rows.append({
            "K": k,
            "PCA log-likelihood": pca_results[k]["log_likelihood"],
            "PCA iterations": pca_results[k]["n_iters"],
            "AE log-likelihood": ae_results[k]["log_likelihood"],
            "AE iterations": ae_results[k]["n_iters"]
        })

    comparison = pd.DataFrame(rows)

    # Sorts the table by k to make the comparison easier to read
    comparison = comparison.sort_values("K").reset_index(drop=True)

    print(title)

    display(
        comparison.style
        .hide(axis="index")
        .format({
            "K": "{:.0f}",
            "PCA log-likelihood": "{:.3f}",
            "PCA iterations": "{:.0f}",
            "AE log-likelihood": "{:.3f}",
            "AE iterations": "{:.0f}"
        })
    )


def compute_silhouette_for_k_range(X, results):
    """
    Computes Silhouette score for clustering results obtained with several values of K.

    Arguments:
        X (pd.DataFrame | np.ndarray): data matrix used for clustering
        results (dict): clustering results indexed by K

    Returns:
        pd.DataFrame: table with K and Silhouette score
    """
    rows = []

    for k in sorted(results.keys()):
        # Get labels for the current value of K
        labels = results[k]["labels"]

        # Compute Silhouette score for the current clustering result
        score = compute_silhouette_score(X, labels)

        rows.append({
            "K": k,
            "silhouette_score": score
        })

        print(f"K={k} | silhouette={score:.4f}")

    silhouette_table = pd.DataFrame(rows)

    return silhouette_table


def build_cluster_size_table(labels, title="Tamaño de clusters"):
    """
    Builds a table with the number and percentage of samples in each cluster.

    Arguments:
        labels (np.ndarray): cluster labels
        title (str): title displayed above the table

    Returns:
        pd.DataFrame: cluster size table
    """
    labels = np.asarray(labels)

    cluster_counts = pd.Series(labels).value_counts().sort_index()

    cluster_size_table = pd.DataFrame({
        "cluster": cluster_counts.index,
        "count": cluster_counts.values,
        "percentage": cluster_counts.values / len(labels) * 100
    })

    print(title)

    display(
        cluster_size_table.style
        .hide(axis="index")
        .format({
            "cluster": "{:.0f}",
            "count": "{:.0f}",
            "percentage": "{:.3f}%"
        })
    )

    return cluster_size_table


def build_cluster_class_table(labels, y_true, class_names=CLASS_NAMES, title="Composición de clases por cluster"):
    """
    Builds a contingency table between clusters and true classes.

    Arguments:
        labels (np.ndarray): cluster labels
        y_true (pd.Series | np.ndarray): true labels
        class_names (dict): mapping from numeric labels to class names
        title (str): title displayed above the table

    Returns:
        pd.DataFrame: cluster-class count table
    """
    labels = np.asarray(labels)
    y_values = y_true.to_numpy() if hasattr(y_true, "to_numpy") else np.asarray(y_true)

    cluster_class_table = pd.crosstab(labels, y_values)
    cluster_class_table.index.name = "cluster"

    # Uses readable class names in the columns
    cluster_class_table.columns = [f"{label}: {class_names.get(label, label)}" for label in cluster_class_table.columns]

    print(title)

    display(
        cluster_class_table.style
        .format("{:.0f}")
    )

    return cluster_class_table


def build_cluster_purity_table(labels, y_true, class_names=CLASS_NAMES, title="Pureza de clusters"):
    """
    Builds a cluster purity table.
    Purity measures the proportion of the dominant true class inside each cluster.

    Formula:
        purity_j = max_c n_jc / sum_c n_jc

    Arguments:
        labels (np.ndarray): cluster labels
        y_true (pd.Series | np.ndarray): true labels
        class_names (dict): mapping from numeric labels to class names
        title (str): title displayed above the table

    Returns:
        pd.DataFrame: cluster purity table
    """
    labels = np.asarray(labels)
    y_values = y_true.to_numpy() if hasattr(y_true, "to_numpy") else np.asarray(y_true)

    rows = []

    for cluster in sorted(np.unique(labels)):
        # Selects true labels assigned to the current cluster
        cluster_classes = y_values[labels == cluster]

        class_counts = pd.Series(cluster_classes).value_counts()
        dominant_class = class_counts.idxmax()
        dominant_count = class_counts.max()
        cluster_size = len(cluster_classes)

        rows.append({
            "cluster": cluster,
            "size": cluster_size,
            "dominant_class": f"{dominant_class}: {class_names.get(dominant_class, dominant_class)}",
            "dominant_count": dominant_count,
            "purity": dominant_count / cluster_size
        })

    purity_table = pd.DataFrame(rows)

    print(title)

    display(
        purity_table.style
        .hide(axis="index")
        .format({
            "cluster": "{:.0f}",
            "size": "{:.0f}",
            "dominant_count": "{:.0f}",
            "purity": "{:.3f}"
        })
    )

    return purity_table
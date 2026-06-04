import numpy as np

def choose_m_components(S, variance_threshold = 0.95):
    """
    Chooses the minimum number of principal components needed to reach a given explained variance threshold.
    If X = U S V^T, then the variance explained by component j is proportional to s_j^2.
    The selected m is the smallest value such that sum_{j=1}^m s_j^2 / sum_{j=1}^p s_j^2 >= variance_threshold.

    Arguments:
        S (np.ndarray): singular values obtained from SVD
        variance_threshold (float): minimum cumulative explained variance to preserve

    Returns:
        int: number of components needed to reach the variance threshold
    """
    # Compute the total variance explained by all singular values
    total_variance = np.sum(S**2)

    # Compute the cumulative proportion of explained variance
    cumulative_variance = np.cumsum(S**2) / total_variance

    # Finds the first component where cumulative variance reaches the threshold
    m_components = np.searchsorted(cumulative_variance, variance_threshold) + 1 # +1 because it returns an index

    return m_components


def fit_pca(X, variance_threshold = 0.95, m_components = None):
    """
    Fits PCA using Singular Value Decomposition.
    First, the data is centered as X_centered = X - mean(X).
    Then, SVD is computed as X_centered = U S V^T.
    The principal components are the rows of V^T, and the transformed data is Z = X_centered V, where V contains the selected components.

    Arguments:
        X (pd.DataFrame | np.ndarray): standardized training data
        variance_threshold (float): minimum cumulative explained variance to preserve when m_components is None
        m_components (int | None): number of principal components to keep

    Returns:
        tuple[np.ndarray, dict]: transformed training data and fitted PCA parameters
    """
    # Converts X to NumPy if it comes as a pandas DataFrame.
    X_values = X.to_numpy() if hasattr(X, "to_numpy") else np.asarray(X)

    # Center the data
    X_mean = X_values.mean(axis=0)
    X_centered = X_values - X_mean

    # Compute SVD
    U, S, V_t = np.linalg.svd(X_centered, full_matrices = False)

    # Chooses the number of components
    if m_components is None:
        m_components = choose_m_components(S, variance_threshold)

    # Keep only the selected principal components
    V_t_m = V_t[:m_components, :]

    # Project the centered data onto the selected principal components
    Z = X_centered @ V_t_m.T

    # Compute explained variance ratios from singular values
    explained_variance_ratio = (S**2) / np.sum(S**2)
    cumulative_variance_ratio = np.cumsum(explained_variance_ratio)

    pca_params = {
        "mean": X_mean,
        "components": V_t_m,
        "singular_values": S,
        "m_components": m_components,
        "explained_variance_ratio": explained_variance_ratio,
        "cumulative_variance_ratio": cumulative_variance_ratio
    }

    return Z, pca_params


def transform_pca(X, pca_params):
    """
    Applies a previously fitted PCA transformation to new data.
    The new data is centered using the training mean and projected as Z = (X - mean_train) V.

    Arguments:
        X (pd.DataFrame | np.ndarray): standardized data to transform
        pca_params (dict): fitted PCA parameters returned by fit_pca

    Returns:
        np.ndarray: transformed data in the PCA space
    """
    # Converts X to NumPy if it comes as a pandas DataFrame
    X_values = X.to_numpy() if hasattr(X, "to_numpy") else np.asarray(X)

    # Centers the data using the training mean learned during fit
    X_centered = X_values - pca_params["mean"]

    # Projects the data onto the principal components learned from training.
    Z = X_centered @ pca_params["components"].T

    return Z


def reconstruct_pca(Z, pca_params):
    """
    Reconstructs data from its PCA representation.
    If PCA transformed the data as Z = (X - mean_train) V, then the approximate reconstruction is X_hat = Z V^T + mean_train.

    Arguments:
        Z (np.ndarray): data projected into PCA space
        pca_params (dict): fitted PCA parameters returned by fit_pca

    Returns:
        np.ndarray: reconstructed data in the standardized feature space
    """
    # Get the principal components used during the PCA projection
    components = pca_params["components"]

    # Reconstruct the centered data and then adds back the PCA training mean
    X_reconstructed = Z @ components + pca_params["mean"]

    return X_reconstructed
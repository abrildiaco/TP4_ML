import numpy as np
import pandas as pd


# K Means -------------------------------------------------------------------------------------------------------------------
def _compute_squared_distances(X, centroids):
    """
    Computes squared Euclidean distances between each observation and each centroid.

    Arguments:
        X (np.ndarray): data matrix
        centroids (np.ndarray): centroid matrix

    Returns:
        np.ndarray: squared distances with shape (n_samples, k)
    """
    # Create pairwise differences between each point and each centroid
    diff = X[:, None, :] - centroids[None, :, :]

    # Sum squared differences across features
    squared_distances = np.sum(diff**2, axis=2)

    return squared_distances


def _initialize_centroids_random(X, k, random_state = 42):
    """
    Initializes k centroids by randomly selecting observations from the dataset.

    Arguments:
        X (np.ndarray): data matrix
        k (int): number of clusters
        random_state (int): random seed for reproducibility

    Returns:
        np.ndarray: initial centroids
    """
    rng = np.random.default_rng(random_state)
    n_samples = X.shape[0]

    # Select k different observations as initial centroids
    centroid_indices = rng.choice(n_samples, size = k, replace = False)

    return X[centroid_indices].copy()


def _initialize_centroids_kmeans_plus_plus(X, k, random_state = 42):
    """
    Initializes k centroids using the k-means++ strategy.
    The first centroid is selected randomly, and each next centroid is selected with probability proportional to its squared distance from the closest existing centroid.

    Arguments:
        X (np.ndarray): data matrix
        k (int): number of clusters
        random_state (int): random seed for reproducibility

    Returns:
        np.ndarray: initial centroids
    """
    rng = np.random.default_rng(random_state)
    n_samples = X.shape[0]

    centroids = []

    # Select the first centroid randomly
    first_index = rng.integers(0, n_samples)
    centroids.append(X[first_index])

    for _ in range(1, k):
        current_centroids = np.array(centroids)

        # Compute the distance from each point to its closest selected centroid
        squared_distances = _compute_squared_distances(X, current_centroids)
        min_squared_distances = np.min(squared_distances, axis=1)

        # Select the next centroid with probability proportional to squared distance
        probabilities = min_squared_distances / np.sum(min_squared_distances)
        next_index = rng.choice(n_samples, p = probabilities)

        centroids.append(X[next_index])

    return np.array(centroids)


def k_means(X, k, max_iters = 100, tol = 1e-4, init = "kmeans++", random_state = 42):
    """
    Runs the k-means clustering algorithm.
    The algorithm alternates between assigning points to the nearest centroid and updating each centroid as the mean of its assigned points.

    Arguments:
        X (pd.DataFrame | np.ndarray): data matrix
        k (int): number of clusters
        max_iters (int): maximum number of iterations
        tol (float): convergence tolerance for centroid movement
        init (str): centroid initialization method, either "random" or "kmeans++"
        random_state (int): random seed for reproducibility

    Returns:
        dict: clustering results with labels, centroids, inertia and number of iterations
    """
    X_values = X.to_numpy() if hasattr(X, "to_numpy") else np.asarray(X)

    rng = np.random.default_rng(random_state)
    n_samples = X_values.shape[0]

    # Select k initial centroids
    if init == "random":
        centroids = _initialize_centroids_random(X_values, k, random_state=random_state)
    elif init == "kmeans++":
        centroids = _initialize_centroids_kmeans_plus_plus(X_values, k, random_state=random_state)
    else:
        raise ValueError("init must be either 'random' or 'kmeans++'.")

    for iter in range(max_iters):
        # Assignment step: assign each point to the nearest centroid
        squared_distances = _compute_squared_distances(X_values, centroids)

        # labels[i] is equivalent to the cluster j where r_ij = 1
        labels = np.argmin(squared_distances, axis=1)

        # Update step: recompute centroids
        new_centroids = []

        for cluster_id in range(k):
            # Select points with r_ij = 1 for the current cluster j
            cluster_points = X_values[labels == cluster_id]

            if len(cluster_points) > 0:
                # Update centroid as mu_j = mean of all points assigned to cluster j
                new_centroids.append(cluster_points.mean(axis=0))
            else:
                # Reinitialize empty clusters with a random observation
                random_index = rng.integers(0, n_samples)
                new_centroids.append(X_values[random_index])

        new_centroids = np.array(new_centroids)

        # Measure how much the centroids moved in this iteration
        centroid_shift = np.linalg.norm(new_centroids - centroids)

        centroids = new_centroids

        if centroid_shift < tol:
            print(f"Convergencia alcanzada en la iteración {iter}")
            break

    # Compute final inertia: sum_i sum_j r_ij ||x_i - mu_j||^2
    final_squared_distances = _compute_squared_distances(X_values, centroids)

    # Select only the distance to the assigned centroid, equivalent to multiplying by r_ij
    final_distances_to_assigned_centroids = final_squared_distances[np.arange(n_samples), labels]

    inertia = np.sum(final_distances_to_assigned_centroids)

    results = {
        "labels": labels,
        "centroids": centroids,
        "inertia": inertia,
        "n_iters": iter + 1
    }

    return results


# GMM --------------------------------------------------------------------------------------------------
def _initialize_gmm_parameters(X, k, random_state = 42, epsilon = 1e-6):
    """
    Initializes the parameters of a Gaussian Mixture Model with diagonal covariance matrices.

    Arguments:
        X (pd.DataFrame | np.ndarray): data matrix
        k (int): number of Gaussian components
        random_state (int): random seed for reproducibility
        epsilon (float): small value added to variances for numerical stability

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]: initial means, weights and diagonal variances
    """
    n_samples= X.shape[0]

    rng = np.random.default_rng(random_state)

    # Choose k random samples as initial means mu_j
    mean_indices = rng.choice(n_samples, size = k, replace = False)
    means = X[mean_indices].copy()

    # Initialize all mixture weights equally
    weights = np.ones(k) / k

    # Use the global variance of each feature as the initial diagonal variance
    global_variance = X.var(axis = 0) + epsilon
    # Repeat the same initial variance vector for each Gaussian component
    variances = np.tile(global_variance, (k, 1))

    return means, variances, weights

def _diagonal_gaussian_log_pdf(X, mean, variance):
    """
    Computes the log probability density of a diagonal multivariate Gaussian distribution.

    Formula:
        log N(x_i | mu_j, Sigma_j) =
        -0.5 * sum_m [log(2*pi*sigma_jm^2) + ((x_im - mu_jm)^2 / sigma_jm^2)]

    Arguments:
        X (pd.DataFrame | np.ndarray): data matrix
        mean (np.ndarray): mean vector of one Gaussian component
        variance (np.ndarray): diagonal variance vector of one Gaussian component

    Returns:
        np.ndarray: log probability density for each observation
    """
    X_values = X.to_numpy() if hasattr(X, "to_numpy") else np.asarray(X)

    # Compute the difference between each point and the component mean
    diff = X_values - mean

    # Compute log-density in a numerically stable way
    log_pdf = -0.5 * np.sum(np.log(2 * np.pi * variance) + (diff**2) / variance, axis=1)

    return log_pdf


def _compute_responsibilities(X, k, means, variances, weights, epsilon=1e-12):
    """
    Computes the responsibility matrix for the E-step of GMM.
    Each responsibility r_ij represents how much observation i belongs to component j.

    Formula:
        r_ij =
        pi_j * N(x_i | mu_j, Sigma_j)
        / sum_l pi_l * N(x_i | mu_l, Sigma_l)

    Arguments:
        X (pd.DataFrame | np.ndarray): data matrix
        k (int): number of Gaussian components
        means (np.ndarray): component means
        variances (np.ndarray): component diagonal variances
        weights (np.ndarray): component weights
        epsilon (float): small value added to avoid division by zero

    Returns:
        np.ndarray: responsibility matrix with shape (n_samples, k)
    """
    X_values = X.to_numpy() if hasattr(X, "to_numpy") else np.asarray(X)

    n_samples = X_values.shape[0]

    log_weighted_densities = np.zeros((n_samples, k))

    for j in range(k):
        # Compute log pi_j + log N(x_i | mu_j, Sigma_j)
        log_pdf_j = _diagonal_gaussian_log_pdf(X_values, means[j], variances[j])
        log_weighted_densities[:, j] = np.log(weights[j] + epsilon) + log_pdf_j

    # Stabilizes exponentials by subtracting the row maximum
    row_max = np.max(log_weighted_densities, axis=1, keepdims=True)
    stabilized = log_weighted_densities - row_max

    # Convert stable log-values back to probabilities
    weighted_densities = np.exp(stabilized)

    # Normalize each row so responsibilities sum to 1
    responsibilities = weighted_densities / (np.sum(weighted_densities, axis=1, keepdims=True) + epsilon)

    return responsibilities


def _update_parameters(X, k, responsibilities, epsilon = 1e-6):
    """
    Updates GMM parameters in the M-step using the responsibility matrix.

    Formula:
        N_j = sum_i r_ij

        pi_j = N_j / n

        mu_j = (1 / N_j) * sum_i r_ij * x_i

        sigma_j^2 = (1 / N_j) * sum_i r_ij * (x_i - mu_j)^2

    Arguments:
        X (pd.DataFrame | np.ndarray): data matrix
        k (int): number of Gaussian components
        responsibilities (np.ndarray): responsibility matrix with shape (n_samples, k)
        epsilon (float): small value added to variances for numerical stability

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]: updated means, variances and weights
    """
    n_samples = X.shape[0]

    # Calculate N_j = sum_i r_ij
    N_j = np.sum(responsibilities, axis=0)

    # Update mixture weights pi_j = N_j / n
    weights = N_j / n_samples

    means = []
    variances = []

    for j in range(k):
        # Select responsibilities for component j
        r_j = responsibilities[:, j]

        # Update mean mu_j = sum_i r_ij x_i / N_j
        mean_j = np.sum(r_j[:, None] * X, axis=0) / N_j[j]
        means.append(mean_j)

        # Update diagonal variance sigma_j^2 = sum_i r_ij (x_i - mu_j)^2 / N_j
        variance_j = np.sum(r_j[:, None] * (X - mean_j) ** 2, axis=0) / N_j[j] + epsilon
        variances.append(variance_j)

    means = np.array(means)
    variances = np.array(variances)

    return means, variances, weights


def _compute_log_likelihood(X, k, means, variances, weights, epsilon=1e-12):
    """
    Computes the log-likelihood of a Gaussian Mixture Model.

    Formula:
        log L = sum_i log(sum_j pi_j * N(x_i | mu_j, Sigma_j))

    Arguments:
        X (pd.DataFrame | np.ndarray): data matrix
        k (int): number of Gaussian components
        means (np.ndarray): component means
        variances (np.ndarray): component diagonal variances
        weights (np.ndarray): component weights
        epsilon (float): small value added to avoid log(0)

    Returns:
        float: log-likelihood value
    """
    X_values = X.to_numpy() if hasattr(X, "to_numpy") else np.asarray(X)

    n_samples = X_values.shape[0]

    log_weighted_densities = np.zeros((n_samples, k))

    for j in range(k):
        # Compute log pi_j + log N(x_i | mu_j, Sigma_j)
        log_pdf_j = _diagonal_gaussian_log_pdf(X_values, means[j], variances[j])
        log_weighted_densities[:, j] = np.log(weights[j] + epsilon) + log_pdf_j

    # Computes log-sum-exp manually for numerical stability
    row_max = np.max(log_weighted_densities, axis = 1, keepdims = True)
    log_total_density = row_max + np.log(np.sum(np.exp(log_weighted_densities - row_max), axis=1, keepdims=True) + epsilon)

    log_likelihood = np.sum(log_total_density)

    return log_likelihood


def gmm(X, k, max_iters=100, tol=1e-10, random_state=42, epsilon=1e-12):
    """
    Runs a Gaussian Mixture Model using the EM algorithm.
    The model uses diagonal covariance matrices for numerical stability and lower computational cost.

    Arguments:
        X (pd.DataFrame | np.ndarray): data matrix
        k (int): number of Gaussian components
        max_iters (int): maximum number of EM iterations
        tol (float): convergence tolerance for log-likelihood improvement
        random_state (int): random seed for reproducibility
        epsilon (float): small value used for numerical stability

    Returns:
        dict: GMM results with labels, responsibilities, means, variances, weights, log-likelihood and number of iterations
    """
    X_values = X.to_numpy() if hasattr(X, "to_numpy") else np.asarray(X)

    means, variances, weights = _initialize_gmm_parameters(X_values, k, random_state=random_state, epsilon=epsilon)

    log_likelihood_prev = -np.inf
    log_likelihood_history = []

    for iter in range(max_iters):
        # E-step: compute responsibilities r_ij
        responsibilities = _compute_responsibilities(X_values, k, means, variances, weights, epsilon=epsilon)

        # M-step: update means, variances and weights
        means, variances, weights = _update_parameters(X_values, k, responsibilities, epsilon=epsilon)

        # Compute current log-likelihood
        log_likelihood = _compute_log_likelihood(X_values, k, means, variances, weights, epsilon=epsilon)
        log_likelihood_history.append(log_likelihood)

        if abs(log_likelihood - log_likelihood_prev) < tol:
            break

        log_likelihood_prev = log_likelihood

    # Assign each point to the component with highest responsibility
    labels = np.argmax(responsibilities, axis = 1)

    results = {
        "labels": labels,
        "responsibilities": responsibilities,
        "means": means,
        "variances": variances,
        "weights": weights,
        "log_likelihood": log_likelihood,
        "log_likelihood_history": log_likelihood_history,
        "n_iters": iter + 1
    }

    return results


# Marginal Gains------------------------------------------------------------------------------------------------------
def compute_marginal_gain(results, metric, mode):
    """
    Computes the marginal gain obtained when increasing K.
    For decreasing metrics such as inertia, gain(K) = metric(K-1) - metric(K).
    For increasing metrics such as log-likelihood, gain(K) = metric(K) - metric(K-1).

    Arguments:
        results (dict): clustering results indexed by K
        metric (str): metric to compare, such as "inertia" or "log_likelihood"
        mode (str): either "decrease" or "increase"

    Returns:
        pd.DataFrame: table with K and marginal gain
    """
    k_sorted = sorted(results.keys())

    rows = []

    for i in range(1, len(k_sorted)):
        k_prev = k_sorted[i - 1]
        k_curr = k_sorted[i]

        # Compute marginal gain for the current metric
        if mode == "decrease":
            gain = results[k_prev][metric] - results[k_curr][metric]
        elif mode == "increase":
            gain = results[k_curr][metric] - results[k_prev][metric]
        else:
            raise ValueError("mode must be either 'decrease' or 'increase'")

        rows.append({
            "K": k_curr,
            "marginal_gain": gain
        })

    marginal_gain_table = pd.DataFrame(rows)

    return marginal_gain_table


# Silhouette score -----------------------------------------------------------------------------------------------------------
def _compute_pairwise_distances(X):
    """
    Computes the Euclidean distance matrix between all pairs of observations.

    Formula:
        D_ij = ||x_i - x_j||^2

    Arguments:
        X (pd.DataFrame | np.ndarray): data matrix

    Returns:
        np.ndarray: pairwise distance matrix
    """
    X_values = X.to_numpy() if hasattr(X, "to_numpy") else np.asarray(X)

    # Compute squared distances using dot products to avoid huge intermediate arrays
    squared_norms = np.sum(X_values**2, axis = 1)
    
    # squared_distances[i, j] = ||x_i - x_j||^2 = ||x_i||^2 + ||x_j||^2 - 2 * x_i . x_j
    squared_distances = squared_norms[:, None] + squared_norms[None, :] - 2 * X_values @ X_values.T

    # Avoid small negative values caused by numerical precision
    squared_distances = np.maximum(squared_distances, 0)

    distances = np.sqrt(squared_distances)

    return distances


def compute_silhouette_score(X, labels):
    """
    Computes the Silhouette score for a clustering result.

    Formula:
        a(i) = average distance from point i to points in its own cluster

        b(i) = minimum average distance from point i to points in another cluster

        s(i) = (b(i) - a(i)) / max(a(i), b(i))

        S = mean_i s(i)

    Arguments:
        X (pd.DataFrame | np.ndarray): data matrix
        labels (np.ndarray): cluster labels

    Returns:
        float: average Silhouette score
    """
    labels = np.asarray(labels)

    distances = _compute_pairwise_distances(X)

    n_samples = distances.shape[0]
    unique_labels = np.unique(labels)

    a_values = np.zeros(n_samples)
    b_values = np.zeros(n_samples)

    for i in range(n_samples):
        # Get the cluster assigned to the current point
        cluster_i = labels[i]

        # Compute a(i): average distance to points in the same cluster
        same_cluster_mask = (labels == cluster_i)
        same_cluster_mask[i] = False # Exclude current point

        if np.sum(same_cluster_mask) > 0:
            a_values[i] = np.mean(distances[i, same_cluster_mask])
        
        else:
            a_values[i] = 0

        # Compute b(i): minimum average distance to points in another cluster
        other_cluster_distances = []

        for cluster_j in unique_labels:
            if cluster_j == cluster_i:
                continue

            other_cluster_mask = labels == cluster_j
            avg_distance = np.mean(distances[i, other_cluster_mask])
            other_cluster_distances.append(avg_distance)

        b_values[i] = np.min(other_cluster_distances)

    # Compute silhouette score for each point
    denominator = np.maximum(a_values, b_values)
    silhouette_values = np.where(denominator > 0, (b_values - a_values) / denominator, 0) # Handle case where a(i) and b(i) are both 0

    silhouette_score = np.mean(silhouette_values)

    return silhouette_score
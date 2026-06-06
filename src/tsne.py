import numpy as np
from src.clustering import _compute_pairwise_distances


def _compute_entropy_and_probabilities(distances_i, beta):
    """
    Computes entropy and conditional probabilities for one observation.

    Formula:
        p_j|i = exp(-beta * D_ij) / sum_k exp(-beta * D_ik)

    Arguments:
        distances_i (np.ndarray): squared distances from one point to all other points
        beta (float): inverse variance parameter

    Returns:
        tuple[float, np.ndarray]: entropy and conditional probabilities
    """
    # Computes unnormalized probabilities
    probabilities = np.exp(-distances_i * beta)

    # Removes self-probability
    probabilities[distances_i == 0] = 0

    sum_probabilities = np.sum(probabilities)

    if sum_probabilities == 0:
        probabilities = np.ones_like(probabilities) / (len(probabilities) - 1)
        probabilities[distances_i == 0] = 0
        sum_probabilities = np.sum(probabilities)

    # Normalizes probabilities
    probabilities = probabilities / sum_probabilities

    # Computes entropy in base 2
    entropy = -np.sum(probabilities * np.log2(probabilities + 1e-12))

    return entropy, probabilities


def _compute_joint_probabilities(X, perplexity = 30, tol = 1e-5, max_iters = 50):
    """
    Computes high-dimensional joint probabilities for t-SNE.

    Formula:
        p_ij = (p_j|i + p_i|j) / (2n)

    Arguments:
        X (pd.DataFrame | np.ndarray): data matrix
        perplexity (float): target perplexity
        tol (float): tolerance for binary search
        max_iters (int): maximum binary search iterations

    Returns:
        np.ndarray: joint probability matrix P
    """
    # t-SNE uses squared Euclidean distances in the high-dimensional space
    distances = _compute_pairwise_distances(X) ** 2
    n_samples = distances.shape[0]

    target_entropy = np.log2(perplexity)
    conditional_probabilities = np.zeros((n_samples, n_samples))

    for i in range(n_samples):
        beta = 1.0
        beta_min = -np.inf
        beta_max = np.inf

        for _ in range(max_iters):
            entropy, probabilities = _compute_entropy_and_probabilities(distances[i], beta)
            entropy_diff = entropy - target_entropy

            if abs(entropy_diff) < tol:
                break

            if entropy_diff > 0:
                beta_min = beta
                beta = beta * 2 if beta_max == np.inf else (beta + beta_max) / 2
            else:
                beta_max = beta
                beta = beta / 2 if beta_min == -np.inf else (beta + beta_min) / 2

        conditional_probabilities[i] = probabilities

    P = (conditional_probabilities + conditional_probabilities.T) / (2 * n_samples)
    P = np.maximum(P, 1e-12)

    return P
    

def _compute_low_dimensional_probabilities(Y):
    """
    Computes low-dimensional probabilities using a Student-t distribution.

    Formula:
        q_ij = (1 + ||y_i - y_j||^2)^(-1) / sum_ab (1 + ||y_a - y_b||^2)^(-1)

    Arguments:
        Y (np.ndarray): low-dimensional coordinates

    Returns:
        tuple[np.ndarray, np.ndarray]: Q matrix and Student-t numerator
    """
    # t-SNE uses squared distances in the low-dimensional Student-t kernel
    squared_distances = _compute_pairwise_distances(Y) ** 2

    numerator = 1 / (1 + squared_distances)
    np.fill_diagonal(numerator, 0)

    Q = numerator / np.sum(numerator)
    Q = np.maximum(Q, 1e-12)

    return Q, numerator


def _compute_tsne_gradient(P, Q, numerator, Y):
    """
    Computes the t-SNE gradient.

    Formula:
        dC/dy_i = 4 * sum_j (p_ij - q_ij) * (1 + ||y_i - y_j||^2)^(-1) * (y_i - y_j)

    Arguments:
        P (np.ndarray): high-dimensional probability matrix
        Q (np.ndarray): low-dimensional probability matrix
        numerator (np.ndarray): Student-t numerator
        Y (np.ndarray): low-dimensional coordinates

    Returns:
        np.ndarray: gradient matrix
    """
    n_samples = Y.shape[0]
    gradient = np.zeros_like(Y)

    PQ_diff = (P - Q) * numerator

    for i in range(n_samples):
        # Computes gradient contribution for point i
        gradient[i] = 4 * np.sum(PQ_diff[:, i][:, None] * (Y[i] - Y), axis=0)

    return gradient


def tsne(X, n_components = 2, perplexity = 30, learning_rate = 100, n_iters = 1000, early_exaggeration = 12, random_state = 42, verbose = True):
    """
    Runs a custom implementation of t-SNE.

    Objective:
        KL(P || Q) = sum_ij p_ij * log(p_ij / q_ij)

    Arguments:
        X (pd.DataFrame | np.ndarray): high-dimensional data
        n_components (int): output dimension
        perplexity (float): target perplexity
        learning_rate (float): gradient descent learning rate
        n_iters (int): number of optimization iterations
        early_exaggeration (float): factor used to separate neighbors at the beginning
        random_state (int): random seed for reproducibility
        verbose (bool): whether to print progress

    Returns:
        np.ndarray: low-dimensional embedding
    """
    rng = np.random.default_rng(random_state)

    X_values = X.to_numpy() if hasattr(X, "to_numpy") else np.asarray(X)
    n_samples = X_values.shape[0]

    P = _compute_joint_probabilities(X_values, perplexity=perplexity)
    P = P * early_exaggeration

    Y = rng.normal(0, 1e-4, size=(n_samples, n_components))

    for iteration in range(n_iters):
        Q, numerator = _compute_low_dimensional_probabilities(Y)
        gradient = _compute_tsne_gradient(P, Q, numerator, Y)

        # Moves in the opposite direction of the gradient to minimize KL divergence
        Y = Y - learning_rate * gradient

        # Keeps the embedding centered around zero
        Y = Y - Y.mean(axis=0)

        if iteration == 250:
            P = P / early_exaggeration

        if verbose and (iteration + 1) % 100 == 0:
            kl_divergence = np.sum(P * np.log(P / Q))
            print(f"Iteration {iteration + 1}/{n_iters} | KL divergence: {kl_divergence:.4f}")

    return Y
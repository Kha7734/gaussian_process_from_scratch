# gp/data/toy_dataset.py

import numpy as np

def generate_toy_data(n_samples: int = 10, noise_std: float = 0.1, random_seed: int = 42):
    """
    Generate toy dataset: y = sin(x) + noise

    Args:
        n_samples (int): Number of samples.
        noise_std (float): Standard deviation of Gaussian noise.
        random_seed (int): Random seed for reproducibility.

    Returns:
        X (np.ndarray): Inputs of shape (n_samples, 1)
        y (np.ndarray): Outputs of shape (n_samples,)
    """
    np.random.seed(random_seed)
    X = np.sort(5 * np.random.rand(n_samples, 1), axis=0)
    y = np.sin(X).ravel() + noise_std * np.random.randn(n_samples)
    return X, y

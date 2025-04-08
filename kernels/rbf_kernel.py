# gp/kernels/rbf_kernel.py

import numpy as np

def rbf_kernel(X1: np.ndarray, X2: np.ndarray, length_scale: float = 1.0, variance: float = 1.0) -> np.ndarray:
    """
    Radial Basis Function (RBF) kernel (also called Squared Exponential kernel).

    Args:
        X1 (np.ndarray): First input set, shape (n_samples_1, n_features)
        X2 (np.ndarray): Second input set, shape (n_samples_2, n_features)
        length_scale (float): Controls smoothness of the function.
        variance (float): Signal variance (amplitude squared).

    Returns:
        np.ndarray: Kernel matrix of shape (n_samples_1, n_samples_2)
    """
    dists = np.sum((X1[:, None, :] - X2[None, :, :])**2, axis=-1)
    return variance * np.exp(-0.5 * dists / length_scale**2)

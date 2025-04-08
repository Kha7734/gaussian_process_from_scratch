# gp/visualizations/plot_gp.py

import numpy as np
import matplotlib.pyplot as plt

def plot_gp(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, mu: np.ndarray, cov: np.ndarray):
    """
    Plot GP mean and uncertainty.

    Args:
        X_train (np.ndarray): Training inputs
        y_train (np.ndarray): Training targets
        X_test (np.ndarray): Test inputs
        mu (np.ndarray): Predictive mean
        cov (np.ndarray): Predictive covariance
    """
    std_dev = np.sqrt(np.diag(cov))

    plt.figure(figsize=(10, 6))
    plt.plot(X_train, y_train, 'kx', mew=2, label='Train Points')
    plt.plot(X_test, mu, 'b', lw=2, label='Predictive Mean')
    plt.fill_between(X_test.ravel(),
                     mu - 2 * std_dev,
                     mu + 2 * std_dev,
                     color='blue',
                     alpha=0.2,
                     label='Confidence Interval (±2σ)')
    plt.title('Gaussian Process Regression')
    plt.legend()
    plt.show()


def plot_gpc(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, prob: np.ndarray):
    """
    Plot GP Classifier mean probability and decision boundary.

    Args:
        X_train (np.ndarray): Training inputs
        y_train (np.ndarray): Training labels (0 or 1)
        X_test (np.ndarray): Test inputs
        prob (np.ndarray): Predictive probabilities for class 1
    """
    plt.figure(figsize=(10, 6))
    plt.plot(X_train[y_train == 0], np.zeros_like(y_train[y_train == 0]), 'ro', label='Class 0')
    plt.plot(X_train[y_train == 1], np.ones_like(y_train[y_train == 1]), 'bo', label='Class 1')
    
    plt.plot(X_test, prob, 'k-', lw=2, label='Predictive Probability')
    plt.axhline(0.5, color='gray', linestyle='--', label='Decision Boundary')
    
    plt.ylim(-0.1, 1.1)
    plt.xlim(X_train.min() - 1, X_train.max() + 1)
    plt.title('Gaussian Process Classification')
    plt.legend()
    plt.show()
import numpy as np
from scipy.special import expit  # Sigmoid function
from scipy.linalg import cho_solve, cho_factor

class SingleTaskGPClassifier:
    def __init__(self, kernel_func, noise=1e-6, max_iter=200):
        """
        Initialize the GP Classifier for a single output.
        
        Args:
            kernel_func: callable, kernel function (e.g., RBF)
            noise: float, jitter for numerical stability
            max_iter: int, maximum number of Laplace optimization iterations
        """
        self.kernel_func = kernel_func
        self.noise = noise
        self.max_iter = max_iter
        self.is_trained = False

    def fit(self, X_train, y_train):
        """
        Fit the GP model to training data using Laplace Approximation.
        """
        self.X_train = X_train
        self.y_train = y_train
        n_samples = X_train.shape[0]

        # Step 1: Compute kernel matrix K
        K = self.kernel_func(X_train, X_train)

        # Step 2: Initialize latent function f = 0
        f = np.zeros(n_samples)

        # Step 3: Optimize using Newton-Raphson iterations
        for iteration in range(self.max_iter):
            pi = expit(f)  # Bernoulli probabilities
            W = np.diag(pi * (1 - pi))  # Weight matrix
            sqrt_W = np.sqrt(W)
            
            # Step 3.1: Build matrix B
            B = np.eye(n_samples) + sqrt_W @ K @ sqrt_W
            L, lower = cho_factor(B + self.noise * np.eye(n_samples))  # Add jitter

            # Step 3.2: Newton-Raphson update step
            b = W @ f + (y_train - pi)
            a = b - sqrt_W @ cho_solve((L, lower), sqrt_W @ (K @ b))
            f = K @ a

            # Optional: Print log likelihood every 5 iterations
            if iteration % 5 == 0:
                ll = self.bernoulli_log_likelihood(f, y_train)
                print(f"Iter {iteration}: Bernoulli Log-Likelihood = {ll:.4f}")

        # Save fitted parameters
        self.f_hat = f
        self.K = K
        self.W = W
        self.L = L
        self.is_trained = True

    def predict(self, X_test):
        """
        Predict probability for new test inputs.
        """
        if not self.is_trained:
            raise RuntimeError("Model must be trained before prediction.")

        # Step 4: Compute predictive distribution
        K_s = self.kernel_func(self.X_train, X_test)
        K_ss = self.kernel_func(X_test, X_test)

        # Mean prediction
        f_mean = K_s.T @ (self.y_train - expit(self.f_hat))

        # Variance prediction
        sqrt_W = np.sqrt(self.W)
        v = cho_solve((self.L, True), sqrt_W @ K_s)
        f_var = np.diag(K_ss) - np.sum(v**2, axis=0)
        f_var = np.clip(f_var, a_min=1e-6, a_max=None)  # Ensure positivity

        # Corrected probability prediction
        gamma = 1.0 / np.sqrt(1.0 + (np.pi * f_var) / 8.0)
        probs = expit(gamma * f_mean)

        return probs  # Shape: (n_test_samples,)

    def bernoulli_log_likelihood(self, f, y):
        """
        Compute the Bernoulli log-likelihood.
        """
        pi = expit(f)
        return np.sum(y * np.log(pi + 1e-6) + (1 - y) * np.log(1 - pi + 1e-6))


# Example kernel function (RBF Kernel)
def rbf_kernel(X1, X2, lengthscale=1.0, outputscale=1.0):
    """
    Radial Basis Function (RBF) kernel.
    """
    dists = np.sum((X1[:, None, :] - X2[None, :, :]) ** 2, axis=-1)
    return outputscale * np.exp(-0.5 / lengthscale**2 * dists)

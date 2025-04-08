import numpy as np
from scipy.special import expit  # sigmoid
from scipy.linalg import cho_solve, cho_factor

class GaussianProcessClassifier:
    def __init__(self, kernel, noise=1e-6, max_iter=10):
        self.kernel = kernel
        self.noise = noise
        self.max_iter = max_iter

    def fit(self, X_train, y_train):
        # Save train set
        self.X_train = X_train
        self.y_train = y_train

        # Step 1: Compute K (n x n) from training inputs
        K = self.kernel(X_train, X_train)

        # Step 2: Initialize f = 0
        f = np.zeros(X_train.shape[0])

        # Step 3: Iterative optimization to find f*
        for i in range(self.max_iter):
            # 3.1: Compute pi = sigmoid(f)
            pi = expit(f)

            # 3.2: Compute W = diag(pi * (1 - pi))
            W = np.diag(pi * (1 - pi))

            # 3.3: Compute Cholesky decomposition of (I + W^{1/2} K W^{1/2})
            sqrt_W = np.sqrt(W)
            B = np.eye(X_train.shape[0]) + sqrt_W @ K @ sqrt_W
            L = np.linalg.cholesky(B + self.noise * np.eye(B.shape[0]))

            # 3.4: Solve for f (Newton-Raphson step)
            b = W @ f + (y_train - pi)
            a = b - sqrt_W @ cho_solve((L, True), sqrt_W @ (K @ b))
            f = K @ a

        self.f = f
        self.K = K
        self.W = W
        self.L = L  # Save L for prediction

    def predict_proba(self, X_test):
        # Step 4: Predictive distribution
        K_s = self.kernel(self.X_train, X_test)
        K_ss = self.kernel(X_test, X_test)

        # Mean prediction
        f_mean = K_s.T @ (self.y_train - expit(self.f))

        # Variance prediction
        v = np.linalg.solve(self.L, np.sqrt(self.W) @ K_s)
        f_var = K_ss - v.T @ v

        # Predictive probabilities
        probs = expit(f_mean / np.sqrt(1 + np.diag(f_var)))
        return probs

    def predict(self, X_test):
        probs = self.predict_proba(X_test)
        return (probs >= 0.5).astype(int)

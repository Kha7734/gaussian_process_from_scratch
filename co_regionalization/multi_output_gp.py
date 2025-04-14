import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.special import expit  # sigmoid

class MultiTaskGPClassifier:
    def __init__(self, coregionalization_kernel, noise=1e-6, max_iter=20):
        """
        coregionalization_kernel: instance của CoregionalizationKernel
        noise: float, jitter để ổn định tính toán
        max_iter: int, số vòng lặp Laplace
        """
        self.kernel = coregionalization_kernel
        self.noise = noise
        self.max_iter = max_iter
        self.is_trained = False

    def fit(self, X_train, task_train, y_train):
        """
        Train Multi-task GP Classification bằng Laplace Approximation.
        
        X_train: (n x d)
        task_train: (n,)
        y_train: (n,)
        """
        n_samples = X_train.shape[0]

        # Step 1: Compute kernel matrix K
        K = self.kernel.compute(X_train, task_train, X_train, task_train)  # (n x n)

        # Step 2: Initialize latent function f = 0
        f = np.zeros(n_samples)

        # Step 3: Laplace Approximation
        for iter in range(self.max_iter):
            pi = expit(f)  # sigmoid(f)
            W_diag = pi * (1 - pi)  # n x 1
            sqrt_W = np.sqrt(W_diag)

            # B = I + sqrt_W @ K @ sqrt_W
            B = np.eye(n_samples) + (sqrt_W[:, None] * K) * sqrt_W[None, :]
            L, lower = cho_factor(B + self.noise * np.eye(n_samples))

            # Newton-Raphson step
            b = W_diag * f + (y_train - pi)
            a = b - sqrt_W * cho_solve((L, lower), sqrt_W * (K @ b))
            f = K @ a

            # Optional: in loss mỗi vài vòng lặp
            if iter % 5 == 0:
                ll = np.sum(y_train * np.log(pi + 1e-6) + (1 - y_train) * np.log(1 - pi + 1e-6))
                print(f"Iteration {iter}: Bernoulli Log-Likelihood = {ll:.4f}")

        # Save fitted params
        self.f_hat = f
        self.X_train = X_train
        self.task_train = task_train
        self.W_diag = W_diag
        self.L = L
        self.lower = lower
        self.y_train = y_train
        self.is_trained = True

    def predict(self, X_test, task_test):
        """
        Predict probability for new samples.
        
        X_test: (n_test x d)
        task_test: (n_test,)
        """
        if not self.is_trained:
            raise RuntimeError("You must train the model first!")

        n_test = X_test.shape[0]

        # Step 1: Compute cross-kernel
        K_s = self.kernel.compute(self.X_train, self.task_train, X_test, task_test)  # (n_train x n_test)

        # Step 2: Compute predictive mean
        f_mean = K_s.T @ (self.y_train - expit(self.f_hat))  # (n_test,)

        # Step 3: Compute predictive variance
        sqrt_W = np.sqrt(self.W_diag)
        v = cho_solve((self.L, self.lower), sqrt_W[:, None] * K_s)
        f_var = np.clip(np.diag(self.kernel.compute(X_test, task_test, X_test, task_test)) - np.sum(v**2, axis=0), a_min=1e-6, a_max=None)

        # Step 4: Predictive probability correction
        gamma = 1.0 / np.sqrt(1.0 + (np.pi * f_var) / 8.0)
        probs = expit(gamma * f_mean)

        return probs  # (n_test,)

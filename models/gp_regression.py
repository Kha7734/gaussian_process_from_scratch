# gp/models/gp_regression.py
import numpy as np
from kernels.rbf_kernel import rbf_kernel
from optimization.hyperparameter_optimization import HyperparameterOptimizer

class GaussianProcessRegressor:
    def __init__(self, length_scale: float = 1.0, variance: float = 1.0, noise: float = 1e-8, optimizer: HyperparameterOptimizer = None):
        """
        Basic Gaussian Process Regressor.

        Args:
            length_scale (float): Length scale for the RBF kernel.
            variance (float): Variance for the RBF kernel.
            noise (float): Noise term added to the diagonal for numerical stability.
            optimizer (HyperparameterOptimizer, optional): Optimizer for hyperparameters.
        """
        self.length_scale = length_scale
        self.variance = variance
        self.noise = noise
        self.optimizer = optimizer

    def fit(self, X_train: np.ndarray, y_train: np.ndarray):
        self.X_train = X_train
        self.y_train = y_train

        # Nếu có optimizer, set model và thực hiện tối ưu trước khi training
        if self.optimizer is not None:
            print("Optimizing hyperparameters...")
            self.optimizer.gp_model = self
            optimized_params = self.optimizer.optimize(X_train, y_train)
            self.length_scale = optimized_params.get('length_scale', self.length_scale)
            self.variance = optimized_params.get('variance', self.variance)
            self.noise = optimized_params.get('noise', self.noise)
            print(f"Optimized parameters: length_scale={self.length_scale:.4f}, "
                f"variance={self.variance:.4f}, noise={self.noise:.6f}")

        # Sau khi (có thể đã) tối ưu, compute lại kernel
        K = rbf_kernel(X_train, X_train, self.length_scale, self.variance)
        self.K = K + self.noise * np.eye(len(X_train))
        
        # Sử dụng SVD để tính inverse thay vì np.linalg.inv trực tiếp
        # hoặc thêm jitter và sử dụng Cholesky
        jitter = 1e-6
        while True:
            try:
                self.K_inv = np.linalg.inv(self.K + jitter * np.eye(len(X_train)))
                break
            except np.linalg.LinAlgError:
                jitter *= 10
                if jitter > 1.0:
                    raise ValueError("Could not compute inverse of kernel matrix. Data might be problematic.")


    def predict(self, X_test: np.ndarray):
        K_s = rbf_kernel(self.X_train, X_test, self.length_scale, self.variance)
        K_ss = rbf_kernel(X_test, X_test, self.length_scale, self.variance) + 1e-8 * np.eye(len(X_test))

        mu_s = K_s.T @ self.K_inv @ self.y_train
        cov_s = K_ss - K_s.T @ self.K_inv @ K_s

        return mu_s, cov_s

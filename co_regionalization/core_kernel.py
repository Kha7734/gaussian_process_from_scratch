import numpy as np

def initialize_random_matrix(num_tasks, rank_R, scale=0.1, seed=None):
    """
    Initialize a random (num_tasks x rank_R) matrix W.
    
    Args:
        num_tasks: int, số lượng tasks (T)
        rank_R: int, số latent factors (R)
        scale: float, hệ số nhân để random nhỏ (tránh khởi tạo quá lớn)
        seed: int, optional random seed for reproducibility

    Returns:
        W: (T x R) numpy array
    """
    if seed is not None:
        np.random.seed(seed)
    
    W = scale * np.random.randn(num_tasks, rank_R)
    return W

def initialize_small_noise_vector(num_tasks, min_value=1e-3, max_value=5e-2, seed=None):
    """
    Initialize a small positive noise vector v.
    
    Args:
        num_tasks: int, số lượng tasks (T)
        min_value: float, giá trị tối thiểu cho noise
        max_value: float, giá trị tối đa cho noise
        seed: int, optional random seed for reproducibility

    Returns:
        v: (T,) numpy array
    """
    if seed is not None:
        np.random.seed(seed)
    
    v = np.random.uniform(min_value, max_value, size=num_tasks)
    return v


class CoregionalizationKernel:
    def __init__(self, base_kernel, num_tasks, rank_R):
        self.base_kernel = base_kernel  # Ví dụ: RBF kernel
        self.W = initialize_random_matrix(num_tasks, rank_R)  # W: (T x R)
        self.v = initialize_small_noise_vector(num_tasks)     # v: (T, )

    def compute(self, X1, tasks1, X2, tasks2):
        """
        X1: (n1 x d) input points
        tasks1: (n1,) task indices
        X2: (n2 x d) input points
        tasks2: (n2,) task indices
        """
        K_input = self.base_kernel(X1, X2)  # (n1 x n2)
        B = self.W @ self.W.T + np.diag(self.v)  # (T x T)

        # Task correlation part
        B_tasks = B[tasks1][:, tasks2]  # (n1 x n2)

        # Final kernel
        return K_input * B_tasks

# from sklearn.gaussian_process.kernels import RBF as rbf_kernel
import numpy as np
from multi_output_gp import MultiTaskGPClassifier
from core_kernel import CoregionalizationKernel
from plot import plot_multitask_gp_decision_boundary

def rbf_kernel(X1, X2, lengthscale=1.0, outputscale=1.0):
    """
    Radial Basis Function (RBF) kernel.
    """
    dists = np.sum((X1[:, None, :] - X2[None, :, :]) ** 2, axis=-1)
    return outputscale * np.exp(-0.5 / lengthscale**2 * dists)

def create_toy_multitask_dataset(n_samples=100, n_tasks=3, seed=None):
    if seed is not None:
        np.random.seed(seed)
    
    X = np.random.randn(n_samples, 2)  # (n_samples, 2 features)
    task_ids = np.random.choice(n_tasks, size=n_samples)  # (n_samples,)
    
    y = (np.sin(X[:, 0] + task_ids) + np.cos(X[:, 1] - task_ids) > 0).astype(int)
    
    return X, task_ids, y

def create_simple_multitask_dataset(n_samples=100, n_tasks=3, seed=None):
    """
    Tạo dataset đơn giản cho multi-task classification:
    - X trong khoảng [-3, 3] (2D)
    - Rule: nếu x1 >=2 hoặc x1 <= -2 thì y=0, còn lại y=1
    - Các task vẫn độc lập assignment.

    Args:
        n_samples: int, số lượng sample
        n_tasks: int, số lượng tasks
        seed: int, random seed

    Returns:
        X: (n_samples, 2)
        task_ids: (n_samples,)
        y: (n_samples,)
    """
    if seed is not None:
        np.random.seed(seed)

    # Step 1: Generate points uniformly from [-3, 3]
    X = np.random.uniform(-3, 3, size=(n_samples, 2))  # (n_samples, 2 features)

    # Step 2: Assign random task ids
    task_ids = np.random.choice(n_tasks, size=n_samples)

    # Step 3: Simple rule based on x1
    y = np.where((X[:, 0] >= 2) | (X[:, 0] <= -2), 0, 1)

    return X, task_ids, y


# 1. Import kernel và model
core_kernel = CoregionalizationKernel(base_kernel=rbf_kernel, num_tasks=2, rank_R=2)
model = MultiTaskGPClassifier(core_kernel, max_iter=100)

# 2. Generate toy data
# X_train, task_train, y_train = create_toy_multitask_dataset(n_samples=500, n_tasks=2, seed=42)
X_train, task_train, y_train = create_simple_multitask_dataset(n_samples=200, n_tasks=2, seed=42)

# 3. Train model
model.fit(X_train, task_train, y_train)

# 4. Predict
probs = model.predict(X_train, task_train)
preds = (probs > 0.5).astype(int)

# 5. Evaluate
acc = np.mean(preds == y_train)
print(f"Training Accuracy on Toy Dataset: {acc:.4f}")

# 6. Plot decision boundary
plot_multitask_gp_decision_boundary(model, X_train, task_train, y_train)
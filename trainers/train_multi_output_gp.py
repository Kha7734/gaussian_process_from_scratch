# trainers/train_multioutput_gp.py

from models.multi_task_classification import MultiOutputGPClassifier

def train_multioutput_gp(X_train, Y_train, num_outputs, kernel_func, noise=1e-6, max_iter=200):
    """
    Train a MultiOutputGPClassifier.

    Args:
        X_train: (n_samples, n_features)
        Y_train: (n_samples, num_outputs)
        num_outputs: int, number of outputs (tasks)
        kernel_func: function, kernel function (shared across outputs)
        noise: float, noise level
        max_iter: int, maximum iterations for Laplace approximation

    Returns:
        model: Trained MultiOutputGPClassifier
    """
    model = MultiOutputGPClassifier(
        num_outputs=num_outputs, 
        kernel_func=kernel_func, 
        noise=noise, 
        max_iter=max_iter
    )
    model.fit(X_train, Y_train)  # ✅ chỉ cần X_train, Y_train
    return model

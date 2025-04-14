import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np

def plot_multitask_gp_decision_boundary(model, X, task_ids, y_true, resolution=100):
    """
    Vẽ decision boundary cho từng task.
    
    Args:
        model: đã train, MultiTaskGPClassifier
        X: (n_samples, 2) features
        task_ids: (n_samples, ) task indices
        y_true: (n_samples, ) true labels
        resolution: độ phân giải lưới vẽ
    """
    num_tasks = np.unique(task_ids).shape[0]
    fig, axes = plt.subplots(1, num_tasks, figsize=(5*num_tasks, 5))

    if num_tasks == 1:
        axes = [axes]  # Ensure list if only 1 task

    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

    xx, yy = np.meshgrid(np.linspace(x_min, x_max, resolution),
                         np.linspace(y_min, y_max, resolution))
    grid = np.c_[xx.ravel(), yy.ravel()]  # (resolution^2, 2)

    for t_idx, task in enumerate(np.unique(task_ids)):
        ax = axes[t_idx]

        # Create grid tasks
        grid_tasks = np.full((grid.shape[0],), task)

        # Predict on grid
        probs = model.predict(grid, grid_tasks)
        Z = probs.reshape(xx.shape)

        # Plot decision boundary
        ax.contourf(xx, yy, Z, levels=[0, 0.5, 1], alpha=0.5, cmap='coolwarm')

        # Plot data points of current task
        mask = (task_ids == task)
        ax.scatter(X[mask, 0], X[mask, 1], c=y_true[mask], cmap=ListedColormap(['blue', 'red']), edgecolors='k')

        ax.set_title(f"Task {task} Decision Boundary")
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_xlabel("Feature 1")
        ax.set_ylabel("Feature 2")

    plt.tight_layout()
    plt.show()


def plot_multitask_gp_decision_boundary_v2(model, X, task_ids, y_true, resolution=100):
    """
    Vẽ decision boundary rõ ràng cho bài toán nhị phân (y=0 hoặc y=1) với Multi-task GP.
    """
    num_tasks = np.unique(task_ids).shape[0]
    fig, axes = plt.subplots(1, num_tasks, figsize=(5*num_tasks, 5))

    if num_tasks == 1:
        axes = [axes]  # Ensure list if only 1 task

    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1


    xx, yy = np.meshgrid(np.linspace(x_min, x_max, resolution),
                         np.linspace(y_min, y_max, resolution))
    grid = np.c_[xx.ravel(), yy.ravel()]

    for t_idx, task in enumerate(np.unique(task_ids)):
        ax = axes[t_idx]

        # Create grid tasks
        grid_tasks = np.full((grid.shape[0],), task)

        # Predict on grid
        probs = model.predict(grid, grid_tasks)
        Z = probs.reshape(xx.shape)

        # Plot decision boundary
        ax.contourf(xx, yy, Z, levels=[0, 0.5, 1], alpha=0.5, cmap='coolwarm')

        # Plot data points for current task
        mask = (task_ids == task)
        X_task = X[mask]
        y_task = y_true[mask]

        # Plot y=0 class
        class0 = (y_task == 0)
        ax.scatter(X_task[class0, 0], X_task[class0, 1],
                   c='blue', marker='o', edgecolors='k', label='Class 0')

        # Plot y=1 class
        class1 = (y_task == 1)
        ax.scatter(X_task[class1, 0], X_task[class1, 1],
                   c='red', marker='^', edgecolors='k', label='Class 1')

        ax.set_title(f"Task {task} Decision Boundary")
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_xlabel("Feature 1")
        ax.set_ylabel("Feature 2")
        ax.legend()

    plt.tight_layout()
    plt.show()

import numpy as np
import matplotlib.pyplot as plt
from trainers.train_multi_output_gp import train_multioutput_gp   # ✅ Đúng module
from predictors.predict_multi_output_gp import predict_multioutput_gp
from models.multi_task_classification import MultiOutputGPClassifier   # ✅ Đúng module
from models.single_task_classification import rbf_kernel

# 1. Tạo toy dataset
np.random.seed(42)
n_samples = 100  # Số lượng mẫu
X_train = np.random.uniform(-3, 3, (n_samples, 1))

# Output 1: Classify if x > 0 (label 1), else 0
y1 = (X_train[:, 0] > 0).astype(int)

# Output 2: Classify if x^2 > 2 (label 1), else 0
y2 = (X_train[:, 0]**2 > 2).astype(int)

Y_train = np.stack([y1, y2], axis=1)  # (n_samples, 2 outputs)

# 2. Tạo test data
X_test = np.linspace(-3.5, 3.5, 100).reshape(-1, 1)

# 3. Train model
model = train_multioutput_gp(
    X_train, Y_train,
    num_outputs=2,
    kernel_func=lambda X1, X2: rbf_kernel(X1, X2, lengthscale=1.0, outputscale=1.0),
    noise=1e-6,
    max_iter=200   # ✅ Không cần truyền tol nữa
)

# 4. Predict
Y_pred_probs = predict_multioutput_gp(model, X_test)  # (100, 2)

# Print min and max of predictions
print("Predicted probabilities for each task:")
print(f"Task 1: Min = {Y_pred_probs[:, 0].min()}, Max = {Y_pred_probs[:, 0].max()}")
print(f"Task 2: Min = {Y_pred_probs[:, 1].min()}, Max = {Y_pred_probs[:, 1].max()}")

# 5. Plot
fig, axes = plt.subplots(2, 1, figsize=(10, 6))

for t in range(2):
    axes[t].plot(X_test[:, 0], Y_pred_probs[:, t], label=f'Task {t+1} Predicted Prob', color='blue')
    axes[t].scatter(X_train[:, 0], Y_train[:, t], label=f'Task {t+1} True Labels', color='red', marker='x')
    axes[t].set_ylim([-0.1, 1.1])
    axes[t].set_ylabel(f'Task {t+1}')
    axes[t].legend()

plt.xlabel('Input X')
plt.suptitle('MultiOutput GP Classification on Toy Dataset')
plt.tight_layout()
plt.show()


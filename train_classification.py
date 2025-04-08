# train_classification.py

import numpy as np
from sklearn.datasets import make_moons
from kernels.rbf_kernel import rbf_kernel
from utils.plotting import plot_gpc
from models.gp_classification import GaussianProcessClassifier  # <-- Import GPC class

# 1. Load dataset
np.random.seed(42)  # reproducible

# Tạo 100 điểm random từ Gaussian
X_train = np.random.randn(100, 1)  # 100 samples, 1 feature

# Gán label: nếu x < 0 thì label 0, ngược lại label 1
y_train = (X_train[:, 0] >= 0).astype(int)

# 2. Initialize and train GPC
gpc = GaussianProcessClassifier(kernel=lambda X1, X2: rbf_kernel(X1, X2, length_scale=1.0, variance=1.0))
gpc.fit(X_train, y_train)

# 3. Predict
X_test = np.linspace(X_train.min() - 1, X_train.max() + 1, 300).reshape(-1, 1)
prob = gpc.predict_proba(X_test)

# 4. Print highest and lowest probabilities
print(f"Highest probability: {np.max(prob)}")
print(f"Lowest probability: {np.min(prob)}")

# 5. Plot
plot_gpc(X_train, y_train, X_test, prob)

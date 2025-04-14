# models/multioutput_gp_classifier.py

import numpy as np
from models.single_task_classification import SingleTaskGPClassifier

class MultiOutputGPClassifier:
    def __init__(self, num_outputs, kernel_func, noise=1e-6, max_iter=200):
        self.num_outputs = num_outputs
        self.models = [SingleTaskGPClassifier(kernel_func, noise=noise, max_iter=max_iter) for _ in range(num_outputs)]

    def fit(self, X_train, Y_train):
        """
        Train multiple GP classifiers independently.
        Args:
            X_train: (n_samples, n_features)
            Y_train: (n_samples, num_outputs)
        """
        assert Y_train.shape[1] == self.num_outputs, "Mismatch between Y_train shape and num_outputs"

        for t in range(self.num_outputs):
            print(f"Training output {t+1}/{self.num_outputs}...")
            self.models[t].fit(X_train, Y_train[:, t])

    def predict(self, X_test):
        """
        Predict multi-output probabilities.
        Args:
            X_test: (n_test_samples, n_features)
        Returns:
            preds: (n_test_samples, num_outputs) probabilities
        """
        all_preds = []
        for t in range(self.num_outputs):
            probs = self.models[t].predict(X_test)
            all_preds.append(probs)

        return np.stack(all_preds, axis=1)  # shape (n_test_samples, num_outputs)

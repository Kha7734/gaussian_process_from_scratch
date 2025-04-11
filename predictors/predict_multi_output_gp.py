import numpy as np

def predict_multioutput_gp(model, X_test):
    """
    Predict multi-output probabilities using a trained MultiOutputGPClassifier.

    Args:
        model: Trained MultiOutputGPClassifier
        X_test: (n_test_samples, n_features)

    Returns:
        preds: (n_test_samples, num_outputs) probabilities
    """
    preds = model.predict(X_test)
    return preds

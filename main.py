# main.py

import numpy as np
from data.toy_dataset import generate_toy_data
from models.gp_regression import GaussianProcessRegressor
from optimization.hyperparameter_optimization import HyperparameterOptimizer
from utils.plotting import plot_gp

def main():
    # Generate data
    X_train, y_train = generate_toy_data(n_samples=10, noise_std=0.1)
    
    # Test points
    X_test = np.linspace(0, 5, 100).reshape(-1, 1)

    # Initialize optimizer
    optimizer = HyperparameterOptimizer(gp_model=None)

    # Train GP
    gp = GaussianProcessRegressor(length_scale=1.0, variance=1.0, noise=1e-5, optimizer=optimizer)
    gp.fit(X_train, y_train)
    
    # Predict
    mu, cov = gp.predict(X_test)

    # Plot
    plot_gp(X_train, y_train, X_test, mu, cov)

if __name__ == "__main__":
    main()

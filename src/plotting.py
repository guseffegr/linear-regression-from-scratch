"""
plotting.py

Utility functions for visualizing data and model behavior.
"""
import matplotlib.pyplot as plt

def plot_feature_vs_target(X, y, name):
    """ Plot a scatter chart of a single feature against the target variable. """
    plt.figure(figsize=(5, 4))
    plt.scatter(X, y, alpha=0.3)
    plt.xlabel(name)
    plt.ylabel("MedHouseVal")
    plt.title(f"{name} vs target")
    plt.show()

def plot_predicted_vs_actual(y, y_pred):
    plt.figure(figsize=(5, 4))
    plt.scatter(y, y_pred, alpha=0.3)
    plt.plot([y.min(), y.max()],
            [y.min(), y.max()],
            color="red", linestyle="--")

    plt.xlabel("Actual values")
    plt.ylabel("Predicted values")
    plt.title("Predicted vs Actual (Validation set)")
    plt.show()

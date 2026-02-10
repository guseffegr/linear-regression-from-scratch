"""
loss.py

Utility functions for regression metrics.
"""
import numpy as np

def rmse(y_true, y_pred):
    """ Compute Root Mean Squared Error (RMSE). """
    m = y_true.shape[0]
    return np.sqrt((1 / m) * np.sum((y_pred - y_true) ** 2))
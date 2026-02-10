"""
model.py

Linear regression implementation from scratch.

Provides a simple linear regression model trained with batch gradient descent.
Supports optional L2 regularization (Ridge) to reduce overfitting.
"""
import numpy as np

class LinearRegressionScratch:
    """ Linear regression model trained using gradient descent. """
    def __init__(self, alpha = 0.01, num_iters = 1000, l2_lambda=0.0):
        """
        Parameters
        ----------
        alpha : float
            Learning rate.
        num_iters : int
            Number of gradient descent iterations.
        l2_lambda : float
            L2 regularization strength (0 disables regularization).
        """
        self.w = None
        self.b = None
        self.alpha = alpha
        self.num_iters = num_iters
        self.cost_history = []
        self.l2_lambda = l2_lambda

    def fit(self, X, y):
        """ Train model parameters on the given training data. """
        m, n = X.shape

        self.w = np.zeros(n) # Initialize weights to zero
        self.b = 0.0 # Initialize bias
        self.cost_history = []

        for _ in range(self.num_iters):
            y_hat = X @ self.w + self.b # Linear model prediction

            d_w = (1 / m) * X.T @ (y_hat - y) # Compute gradients of MSE loss
            d_b = (1 / m) * np.sum(y_hat - y)
            
            if self.l2_lambda > 0:
                d_w += (self.l2_lambda / m) * self.w

            self.w -= self.alpha * d_w # Gradient descent update
            self.b -= self.alpha * d_b

            cost = (1 / (m * 2)) * np.sum((y_hat - y) ** 2)

            if self.l2_lambda > 0: # Add L2 regularization to weight gradient (does not apply to bias)
                cost += (self.l2_lambda / (2 * m)) * np.sum(self.w ** 2)
                
            self.cost_history.append(cost)
        
    def predict(self, X):
        """ Predict target values for given input features. """
        assert self.w is not None and self.b is not None, "Model is not fitted yet. Call fit() first."
        return X @ self.w + self.b
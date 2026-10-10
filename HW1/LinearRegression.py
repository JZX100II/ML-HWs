import numpy as np
import torch

class LinearRegression:
    def __init__(self, method='gradient_descent'):
        self.method = method
        self.W = None
        self.b = None
        self.loss_history = []

    def fit(self, X, y, epochs=1000, lr=0.01):
        n_samples, n_features = X.shape
        self.W = np.zeros((n_features, 1))
        self.b = 0.0
        y = y.reshape(-1, 1)

        if self.method == 'gradient_descent':
            self._fit_gradient_descent(X, y, epochs, lr)
        elif self.method == 'normal_equation':
            self._fit_normal_equation(X, y)
        else:
            raise ValueError("Method must be 'gradient_descent' or 'normal_equation'.")

    def _fit_gradient_descent(self, X, y, epochs, lr):
        n = len(y)
        self.loss_history = []

        for epoch in range(epochs):
            # Forward pass: predict
            y_pred = X @ self.W + self.b
            
            # Calculate error
            error = y_pred - y
            
            # MSE Loss: (1/n) * sum((y_pred - y)^2)
            loss = np.mean(error ** 2)
            self.loss_history.append(loss)
            
            # Gradients
            dW = (2/n) * (X.T @ error)
            db = (2/n) * np.sum(error)
            
            # Update parameters using learning rate
            self.W -= lr * dW
            self.b -= lr * db

            if epoch % 200 == 0:
                print(f"Epoch {epoch}: Loss = {loss:.4f}")

    def _fit_normal_equation(self, X, y):
        """
        θ = (X^T * X)^-1 * X^T * y
        """
        # Add bias column (ones) to X
        X_ones = np.c_[np.ones((X.shape[0], 1)), X]
        
        # Convert to torch for pinv
        X_torch = torch.tensor(X_ones, dtype=torch.float32)
        y_torch = torch.tensor(y, dtype=torch.float32)

        # Normal equation using pseudoinverse
        theta = torch.linalg.pinv(X_torch) @ y_torch

        # Extract bias and weights
        self.b = theta[0].item()
        self.W = theta[1:].detach().numpy().reshape(-1, 1)

    def predict(self, X):
        return (X @ self.W + self.b).flatten()
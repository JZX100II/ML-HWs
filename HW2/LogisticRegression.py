import numpy as np

class LogisticRegression:
    def __init__(self, learning_rate=0.1, epochs=1000, class_weight=None, verbose=False):
        self.lr = learning_rate
        self.epochs = epochs
        self.class_weight = class_weight
        self.verbose = verbose
        self.weights = None
        self.bias = 0.0
        self.loss_history = []

    def _sigmoid(self, z):
        return 1.0 / (1.0 + np.exp(-z))

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0.0

        # Compute class weights
        if self.class_weight == 'balanced':
            n_pos = np.sum(y == 1)
            n_neg = np.sum(y == 0)
            w1 = n_neg / n_pos  # higher weight for minority class
            w0 = 1.0
            sample_weights = np.where(y == 1, w1, w0)
        elif isinstance(self.class_weight, dict):
            w0 = self.class_weight.get(0, 1.0)
            w1 = self.class_weight.get(1, 1.0)
            sample_weights = np.where(y == 1, w1, w0)
        else:
            sample_weights = np.ones_like(y)

        for epoch in range(self.epochs):
            # Linear combination
            z = np.dot(X, self.weights) + self.bias
            preds = self._sigmoid(z)

            # Weighted binary cross‑entropy loss
            eps = 1e-15
            loss = -np.mean(sample_weights * (y * np.log(preds + eps) + (1 - y) * np.log(1 - preds + eps)))
            self.loss_history.append(loss)

            # Gradients
            dz = (preds - y) * sample_weights
            dw = np.dot(X.T, dz) / n_samples
            db = np.mean(dz)

            self.weights -= self.lr * dw
            self.bias -= self.lr * db

            if self.verbose and epoch % 100 == 0:
                print(f"Epoch {epoch}, Loss: {loss:.6f}")

    def predict_proba(self, X):
        z = np.dot(X, self.weights) + self.bias
        return self._sigmoid(z)

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) >= threshold).astype(int)
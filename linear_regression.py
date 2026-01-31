import numpy as np


class LinearRegression:
    def __init__(self, lr=0.01, n_iter=1000, weights=0.0, bias=0.0):
        self.lr = lr
        self.n_iter = n_iter
        self.weights = weights
        self.bias = bias

    def fit(self, X, y):
        n_samples, n_features = X.shape

        self.weights = np.full(n_features, self.weights)

        for _ in range(self.n_iter):
            y_hat = np.dot(X, self.weights) + self.bias

            loss = y_hat - y
            dw = np.dot(X.T, loss) / n_samples
            db = np.sum(loss) / n_samples

            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    def predict(self, X):
        return np.dot(X, self.weights) + self.bias

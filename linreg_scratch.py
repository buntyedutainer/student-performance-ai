# Cell 11: save your model so later days can import it
import numpy as np

def gradient_descent(X, y, lr=0.05, epochs=3000):
    w, b, history = np.zeros(X.shape[1]), 0.0, []
    n = len(y)
    for _ in range(epochs):
        error = X @ w + b - y
        w -= lr * (2 / n) * (X.T @ error)
        b -= lr * (2 / n) * error.sum()
        history.append(np.mean(error ** 2))
    return w, b, history

class LinearRegressionScratch:
    def __init__(self, lr=0.05, epochs=3000):
        self.lr, self.epochs = lr, epochs

    def fit(self, X, y):
        X, y = np.asarray(X, float), np.asarray(y, float)
        self.mu, self.sigma = X.mean(axis=0), X.std(axis=0)
        self.sigma[self.sigma == 0] = 1.0
        self.w, self.b, self.history = gradient_descent((X - self.mu) / self.sigma, y, self.lr, self.epochs)
        return self

    def predict(self, X):
        return ((np.asarray(X, float) - self.mu) / self.sigma) @ self.w + self.b

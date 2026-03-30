import numpy as np

def sigmoid(x):
    x = np.clip(x, -50, 50)
    return 1 / (1 + np.exp(-x))

class TinyModel:
    def __init__(self, input_dim):
        self.weights = np.zeros(input_dim)
        self.bias = 0.0

    def predict(self, X):
        return sigmoid(np.dot(X, self.weights) + self.bias)

    def train(self, X, y, lr, epochs):
        for _ in range(epochs):
            preds = self.predict(X)
            error = preds - y
            self.weights -= lr * np.dot(X.T, error) / len(X)
            self.bias -= lr * np.mean(error)

    def get_params(self):
        return [self.weights, self.bias]

    def set_params(self, params):
        self.weights, self.bias = params

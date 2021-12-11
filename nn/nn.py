import numpy as np


class SimpleNet:
    def __init__(self):
        pass

    def sigmoid(self, x: np.array) -> np.array:
        return 1.0 / (1.0 + np.exp(-x))

    def sigmoid_deriv(self, x: np.array) -> np.array:
        s = self.sigmoid(x)
        return s * (1.0 - s)

    def softmax(self, x: np.array) -> np.array:
        s = np.exp(x) / np.max(x)
        s /= s.sum(axis=0)
        return s

    def softmax_deriv(self, x: np.array) -> np.array:
        s = self.softmax(x)
        return s * (1.0 - s)

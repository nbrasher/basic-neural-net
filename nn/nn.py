from time import time
from typing import List

import numpy as np


class SimpleNet:
    """Simple dense neural network.
    Uses sigmoid activation function for first N - 1 layers, softmax for output.
    """

    def __init__(self, sizes: List[int]):
        self.sizes = sizes

        # Random initialization for weight matrices
        self.W = [
            np.random.randn(sizes[i + 1], sizes[i]) / np.sqrt(1 / sizes[i + 1])
            for i in range(len(sizes) - 1)
        ]
        self.A = [None for _ in range(len(sizes) - 1)]
        self.Z = [None for _ in range(len(sizes) - 1)]
        self.params = {}

    def sigmoid(self, x: np.array) -> np.array:
        return np.divide(1.0, 1.0 + np.exp(-x))

    def sigmoid_deriv(self, x: np.array) -> np.array:
        s = self.sigmoid(x)
        return s * (1.0 - s)

    def softmax(self, x: np.array) -> np.array:
        s = np.exp(x) / np.max(x)
        s = np.divide(s, s.sum(axis=0))
        return s

    def softmax_deriv(self, x: np.array) -> np.array:
        s = self.softmax(x)
        return s * (1.0 - s)

    def forward_pass(self, x: np.array) -> np.array:
        self.A[0] = x
        ret = None

        for i in range(len(self.sizes) - 1):
            self.Z[i] = self.W[i] @ self.A[i]

            if i == (len(self.sizes) - 2):
                ret = self.softmax(self.Z[i])
            else:
                self.A[i + 1] = self.sigmoid(self.Z[i])

        return ret

    def backward_pass(self, y, output):
        delta = [None for _ in range(len(self.sizes) - 1)]

        error = 2 * (output - y) / output.shape[0] * self.softmax_deriv(self.Z[-1])

        for i in range(len(self.sizes) - 2, -1, -1):
            if i != len(self.sizes) - 2:
                error = (self.W[i + 1].T @ error) * self.sigmoid_deriv(self.Z[i])

            delta[i] = np.outer(error, self.A[i])

        return delta

    def compute_accuracy(self, x_val, y_val):
        predictions = []

        for x, y in zip(x_val, y_val):
            output = self.forward_pass(x)
            pred = np.argmax(output)
            predictions.append(pred == np.argmax(y))

        return np.mean(predictions)

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        lr_rate: float = 0.05,
        epochs: int = 100,
    ):

        for e in range(epochs):
            start = time()

            for x, y in zip(X_train, y_train):
                output = self.forward_pass(x)
                delta = self.backward_pass(y, output)

                # Update weights from backwards pass
                for i, grad in enumerate(delta):
                    self.W[i] -= lr_rate * grad

            accuracy = self.compute_accuracy(X_test, y_test)

            print(
                "Epoch: {}, Time Spent: {:.2f}s, Accuracy: {:.2f}%".format(
                    str(e).zfill(3), time() - start, accuracy * 100
                )
            )

from time import time
from typing import List

import numpy as np


class SimpleNet:
    """Simple dense neural network.
    Uses sigmoid activation function for first N - 1 layers, softmax for output.
    """

    def __init__(self, sizes: List[int]):
        self.sizes = sizes

        # Random initialization for weights, using Xavier initialization for stability
        self.W = [
            np.random.normal(
                scale=np.sqrt(2.0 / (sizes[i + 1] + sizes[i])),
                size=(sizes[i + 1], sizes[i]),
            )
            for i in range(len(sizes) - 1)
        ]
        self.A = [None for _ in range(len(sizes) - 1)]
        self.Z = [None for _ in range(len(sizes) - 1)]
        self.params = {}

    def sigmoid(self, x: np.ndarray) -> np.ndarray:
        return np.divide(1.0, 1.0 + np.exp(-x))

    def sigmoid_deriv(self, x: np.ndarray) -> np.ndarray:
        s = self.sigmoid(x)
        return s * (1.0 - s)

    def softmax(self, x: np.ndarray) -> np.ndarray:
        s = np.exp(x - x.max())
        s_sum = s.sum(axis=0)
        s = np.divide(s, s_sum, where=s_sum > 1e-15)
        return s

    def softmax_deriv(self, x: np.ndarray) -> np.ndarray:
        s = self.softmax(x)
        return s * (1.0 - s)

    def forward_pass(self, x: np.ndarray) -> np.ndarray:
        self.A[0] = x
        ret = None

        for i in range(len(self.sizes) - 1):
            self.Z[i] = self.W[i] @ self.A[i]

            if i == (len(self.sizes) - 2):
                ret = self.softmax(self.Z[i])
            else:
                self.A[i + 1] = self.sigmoid(self.Z[i])

        return ret

    def backward_pass(self, y: np.ndarray, output: np.ndarray) -> np.ndarray:
        delta = [None for _ in range(len(self.sizes) - 1)]

        error = 2 * (output - y) / output.shape[0] * self.softmax_deriv(self.Z[-1])

        for i in range(len(self.sizes) - 2, -1, -1):
            if i != len(self.sizes) - 2:
                error = (self.W[i + 1].T @ error) * self.sigmoid_deriv(self.Z[i])

            delta[i] = np.outer(error, self.A[i])

        return delta

    def compute_accuracy(self, x_val: np.ndarray, y_val: np.ndarray) -> float:
        predictions = [
            np.argmax(self.forward_pass(x)) == np.argmax(y)
            for x, y in zip(x_val, y_val)
        ]

        return np.mean(predictions)

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        lr_rate: float = 0.01,
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
                "Epoch: {}, Time: {:.1f}s, Validation accuracy: {:.2f}%".format(
                    str(e).zfill(3), time() - start, accuracy * 100
                )
            )

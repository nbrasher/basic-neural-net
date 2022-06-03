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

    def backward_pass(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        delta = [None for _ in range(len(self.sizes) - 1)]

        # Equaivalent to categorical cross-entropy loss gradient
        error = (y_pred - y_true) / y_pred.shape[0]

        for i in range(len(self.sizes) - 2, -1, -1):
            if i != len(self.sizes) - 2:
                error = (self.W[i + 1].T @ error) * self.sigmoid_deriv(self.Z[i])

            delta[i] = np.outer(error, self.A[i])

        return delta

    @staticmethod
    def accuracy(y_pred: np.ndarray, y_true: np.ndarray) -> float:
        predictions = [np.argmax(yp) == np.argmax(yt) for yp, yt in zip(y_pred, y_true)]

        return np.mean(predictions)

    @staticmethod
    def cross_entropy_loss(y_pred: np.ndarray, y_true: np.ndarray) -> float:
        loss = [-np.dot(np.log(yp), yt) for yp, yt in zip(y_pred, y_true)]

        return np.sum(loss)

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        lr_rate: float = 1e-2,
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

            y_pred_train = [self.forward_pass(x) for x in X_train]
            y_pred_test = [self.forward_pass(x) for x in X_test]

            print("Epoch: {}, Time: {:.1f}s".format(str(e).zfill(3), time() - start))
            print(
                "Train Loss: {:,.1f}, Train Accuracy: {:.2f}%".format(
                    self.cross_entropy_loss(y_pred_train, y_train),
                    100 * self.accuracy(y_pred_train, y_train),
                )
            )
            print(
                "Val Loss: {:,.1f}, Val Accuracy: {:.2f}%".format(
                    self.cross_entropy_loss(y_pred_test, y_test),
                    100 * self.accuracy(y_pred_test, y_test),
                )
            )

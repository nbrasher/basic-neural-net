from abc import ABC, abstractmethod
from time import time
from typing import List

import numpy as np

from nn.layers import Dense, MultiClassLastLayer
from nn.metrics import accuracy, cross_entropy_loss


class Model(ABC):
    """ Basic neural net model class """
    def __init__(self):
        self.layers = []
    
    def forward_pass(self, x: np.ndarray) -> np.ndarray:
        for layer in self.layers:
            x = layer.forward_pass(x)
        
        return x

    def backward_pass(self, error: np.ndarray, lr: float):
        for layer in self.layers[::-1]:
            error = layer.backward_pass(error, lr)
    
    @abstractmethod
    def train(self):
        pass


class MultiClassDense(Model):
    """ Simple flat, dense neural network for multi-class classification.
        First few layers are dense relu, final layer is a dense softmax layer.
    """

    def __init__(self, input_shape: int, sizes: List[int]):
        super().__init__()
        if len(sizes) > 1:
            for size in sizes[: -1]:
                self.layers.append(Dense(input_shape, size, "relu"))
                input_shape = size
            
        self.layers.append(MultiClassLastLayer(input_shape, sizes[-1]))

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        learning_rate: float = 1e-3,
        epochs: int = 100,
    ):

        for e in range(epochs):
            start = time()

            for x, y in zip(X_train, y_train):
                res = self.forward_pass(x)
                self.backward_pass(res - y, learning_rate)

            y_pred_train = [self.forward_pass(x) for x in X_train]
            y_pred_test = [self.forward_pass(x) for x in X_test]

            print("Epoch: {}, Time: {:.1f}s".format(str(e).zfill(3), time() - start))
            print(
                "Train Loss: {:,.1f}, Train Accuracy: {:.2f}%".format(
                    cross_entropy_loss(y_pred_train, y_train),
                    100 * accuracy(y_pred_train, y_train),
                )
            )
            print(
                "Val Loss: {:,.1f}, Val Accuracy: {:.2f}%".format(
                    cross_entropy_loss(y_pred_test, y_test),
                    100 * accuracy(y_pred_test, y_test),
                )
            )

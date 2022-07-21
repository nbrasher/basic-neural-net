from abc import ABC, abstractmethod
from math import prod
from typing import Tuple

import numpy as np

from nn.activation import relu, relu_deriv, sigmoid, sigmoid_deriv, softmax


class Layer(ABC):
    @abstractmethod
    def forward_pass(self, x: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def backward_pass(self, x: np.ndarray, lr: float) -> np.ndarray:
        pass


class Dense(Layer):
    """Basic flat, dense neural network layer"""

    def __init__(self, input_shape: int, output_shape: int, activation: str):
        self.W = np.random.normal(
            scale=np.sqrt(2.0 / (input_shape + output_shape)),
            size=(output_shape, input_shape),
        )
        self.A = np.empty((input_shape, 1))
        self.Z = np.empty((output_shape, 1))

        if activation == "relu":
            self.act = relu
            self.act_deriv = relu_deriv
        elif activation == "sigmoid":
            self.act = sigmoid
            self.act_deriv = sigmoid_deriv
        elif activation == "softmax":
            self.act = softmax
            self.act_deriv = None  # The softmax gradient is more complex and unneeded
        else:
            raise ValueError(
                "activation fn must be one of ['relu', 'sigmoid'],"
                + f" but {activation} was passed"
            )

    def forward_pass(self, x: np.ndarray) -> np.ndarray:
        self.A = x
        self.Z = self.W @ self.A
        return self.act(self.Z)

    def backward_pass(self, x: np.ndarray, lr: float) -> np.ndarray:
        error = x * self.act_deriv(self.Z)
        ret = self.W.T @ error
        self.W -= lr * np.outer(error, self.A)

        return ret


class MultiClassLastLayer(Dense):
    """Basic flat final multi-class neural network layer.
    Split this one out separately because of the loss function math.
    """

    def __init__(self, input_shape: int, output_shape: int):
        super().__init__(input_shape, output_shape, "softmax")

    def backward_pass(self, x: np.ndarray, lr: float) -> np.ndarray:
        error = x / x.shape[0]
        ret = self.W.T @ error
        self.W -= lr * np.outer(error, self.A)

        return ret


class Flatten(Layer):
    """Translation layer from a 2- or 3-D tensor to a column vector"""

    def __init__(self, input_shape: Tuple[int]):
        self.dim = len(input_shape)
        self.input_shape = input_shape
        self.output_shape = prod(input_shape)

    def forward_pass(self, x: np.ndarray) -> np.ndarray:
        return x.reshape((self.output_shape, 1))

    def backward_pass(self, x: np.ndarray, lr: float) -> np.ndarray:
        return x.reshape(self.input_shape)


class Dropout(Layer):
    """Dropout layer to radomly remove connections and prevent overfitting"""

    def __init__(self, input_shape: int, factor: float):
        if not isinstance(factor, float) or factor < 0.0 or factor > 1.0:
            raise ValueError("factor must be a float between 0. to 1. (inclusive)")
        self.input_shape = input_shape
        self.factor = factor
        self.mask = np.array([])

    def forward_pass(self, x: np.ndarray) -> np.ndarray:
        self.mask = (
            np.random.random_sample((self.input_shape, 1)) > self.factor
        ).astype(int)
        return (x * self.mask) * (1.0 / (1.0 - self.factor))

    def backward_pass(self, x: np.ndarray, lr: float) -> np.ndarray:
        return (x * self.mask) * (1.0 - self.factor)


class Conv2D(Layer):
    """2D Convolutional layer for computer vision"""

    def __init__(
        self, n_kernels: int, kernel_shape: Tuple[int], input_shape: Tuple[int]
    ):
        self.input_shape = input_shape
        self.kernel_shape = kernel_shape
        self.output_shape = (
            input_shape[0] - kernel_shape[0],
            input_shape[1] - kernel_shape[1],
            n_kernels,
        )
        self.W = np.random.normal(
            scale=np.sqrt(2.0 / (prod(self.output_shape) * prod(self.kernel_shape))),
            size=(kernel_shape[0], kernel_shape[1], n_kernels),
        )
        self.A = np.empty(self.output_shape)
        self.Z = np.empty(self.output_shape)

    def forward_pass(self, x: np.ndarray) -> np.ndarray:
        pass

    def backward_pass(self, x: np.ndarray, lr: float) -> np.ndarray:
        pass

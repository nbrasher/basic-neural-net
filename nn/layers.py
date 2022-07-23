from abc import ABC, abstractmethod
from math import prod
from typing import Tuple

import numpy as np
from numpy.random import random_sample
from scipy.signal import convolve

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
        self.mask = (random_sample((self.input_shape, 1)) > self.factor).astype(int)
        return (x * self.mask) * (1.0 / (1.0 - self.factor))

    def backward_pass(self, x: np.ndarray, lr: float) -> np.ndarray:
        return (x * self.mask) * (1.0 - self.factor)


class Conv2D(Layer):
    """2D Convolutional layer for computer vision"""

    def __init__(
        self, n_kernels: int, kernel_shape: Tuple[int], input_shape: Tuple[int]
    ):
        # TODO - check input and output shape dimensions
        self.input_shape = input_shape
        self.kernel_shape = kernel_shape
        self.n_kernels = n_kernels
        self.output_shape = (
            input_shape[0] - kernel_shape[0] + 1,
            input_shape[1] - kernel_shape[1] + 1,
            n_kernels,
        )
        self.W = np.random.normal(
            scale=np.sqrt(2.0 / (prod(self.output_shape) * prod(kernel_shape))),
            size=tuple(list(kernel_shape) + [input_shape[2]] + [n_kernels]),
        )
        self.A = np.empty(self.output_shape)

    def forward_pass(self, x: np.ndarray) -> np.ndarray:
        self.A = x
        cnv_res = [convolve(slice, x, "valid") for slice in np.moveaxis(self.W, -1, 0)]
        self.Z = np.stack(cnv_res, axis=2).reshape(self.output_shape)

        return relu(self.Z)

    def backward_pass(self, x: np.ndarray, lr: float) -> np.ndarray:
        error = x * relu_deriv(self.Z)

        # Propagated error
        ret = np.empty(self.input_shape)
        for i in range(self.input_shape[2]):
            output_layers = np.array(
                [
                    convolve(error[:, :, j], self.W[:, :, i, j], "full")
                    for j in range(self.n_kernels)
                ]
            )
            ret[:, :, i] = output_layers.sum(axis=0)

        # Weight gradient
        W_grad = np.empty(self.W.shape)
        for i in range(self.n_kernels):
            W_grad[:, :, :, i] = convolve(error[:, :, [i]], self.A, "valid")

        self.W -= lr * W_grad

        return ret

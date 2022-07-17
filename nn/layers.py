from abc import ABC, abstractmethod

import numpy as np

from nn.activation import relu, relu_deriv, sigmoid, sigmoid_deriv, softmax


class Layer(ABC):
    def __init__(self, input_shape: int, output_shape: int):
        self.W = np.random.normal(
            scale=np.sqrt(2.0 / (input_shape + output_shape)),
            size=(output_shape, input_shape),
        )
        self.A = np.empty((input_shape, 1))
        self.Z = np.empty((output_shape, 1))
    
    @abstractmethod
    def forward_pass(self, x: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def backward_pass(self, x: np.ndarray, lr: float) -> np.ndarray:
        pass


class Dense(Layer):
    """ Basic flat, dense neural network layer """
    def __init__(self, input_shape: int, output_shape: int, activation: str):
        super().__init__(input_shape, output_shape)

        if activation == "relu":
            self.act = relu
            self.act_deriv = relu_deriv
        elif activation == "sigmoid":
            self.act = sigmoid
            self.act_deriv = sigmoid_deriv
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


class MultiClassLastLayer(Layer):
    """ Basic flat final multi-class neural network layer.
        Split this one out separately because of the loss function math.
    """
    def __init__(self, input_shape: int, output_shape: int):
        super().__init__(input_shape, output_shape)

    def forward_pass(self, x: np.ndarray) -> np.ndarray:
        self.A = x
        return softmax(self.W @ self.A)

    def backward_pass(self, x: np.ndarray, lr: float) -> np.ndarray:
        error = x / x.shape[0]
        ret = self.W.T @ error
        self.W -= lr * np.outer(error, self.A)

        return ret

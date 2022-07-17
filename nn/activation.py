import numpy as np


def sigmoid(x: np.ndarray) -> np.ndarray:
    return np.divide(1.0, 1.0 + np.exp(-x))

def sigmoid_deriv(x: np.ndarray) -> np.ndarray:
    s = sigmoid(x)
    return s * (1.0 - s)

def relu(x: np.ndarray) -> np.ndarray:
    return np.where(x > 0, x, 0)

def relu_deriv(x: np.ndarray) -> np.ndarray:
    return np.where(x > 0, 1, 0)

def softmax(x: np.ndarray) -> np.ndarray:
    s = np.exp(x - x.max())
    s_sum = s.sum(axis=0)
    s = np.divide(s, s_sum, where=s_sum > 1e-15)
    return s

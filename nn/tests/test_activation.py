from math import exp

import numpy as np
from numpy.testing import assert_array_almost_equal

from nn.activation import relu, relu_deriv, sigmoid, sigmoid_deriv, softmax


def test_sigmoid():
    a = np.array(range(5))
    ans = np.array([(1.0 / (1.0 + exp(-i))) for i in range(5)])

    assert_array_almost_equal(sigmoid(a), ans)


def test_sigmoid_deriv():
    a = np.array(range(5))
    ans = np.array([((exp(-i)) / ((1.0 + exp(-i)) ** 2)) for i in range(5)])

    assert_array_almost_equal(sigmoid_deriv(a), ans)


def test_relu():
    a = np.array(range(-3, 4))
    ans = np.array([0, 0, 0, 0, 1, 2, 3])

    assert_array_almost_equal(relu(a), ans)


def test_relu_deriv():
    a = np.array(range(-3, 4))
    ans = np.array([0, 0, 0, 0, 1, 1, 1])

    assert_array_almost_equal(relu_deriv(a), ans)


def test_softmax():
    a = np.array(range(5))
    ans = np.array([exp(i) for i in range(5)])
    ans /= sum(ans)

    assert_array_almost_equal(softmax(a), ans)

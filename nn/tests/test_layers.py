import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal, assert_array_equal

from nn.layers import Conv2D, Dense, Dropout, Flatten


@pytest.fixture
def dense_layer():
    layer = Dense(input_shape=2, output_shape=3, activation="relu")

    # Replace weights for test purposes
    layer.W = np.array([[1.0, 1.0], [1.0, 0.0], [-1.0, -1.0]])
    yield layer


@pytest.fixture
def conv_layer():
    layer = Conv2D(n_kernels=2, kernel_shape=(2, 2), input_shape=(3, 3, 1))

    # Replace weights for test purposes
    layer.W = np.array([[[[1.0, 2.0]], [[1.0, 2.0]]], [[[1.0, 2.0]], [[1.0, 2.0]]]])
    yield layer


def test_dense(dense_layer):
    assert dense_layer.W.shape == (3, 2)
    assert dense_layer.A.shape == (2, 1)
    assert dense_layer.Z.shape == (3, 1)


def test_error():
    # Raises error on incorrect activation fn
    with pytest.raises(ValueError):
        Dense(input_shape=2, output_shape=4, activation="foo")


def test_dense_fp(dense_layer):
    input = np.array([[4.0], [2.0]])
    res = dense_layer.forward_pass(input)
    assert_array_almost_equal(dense_layer.A, input)
    assert_array_almost_equal(dense_layer.Z, np.array([[6], [4], [-6]]))
    assert_array_almost_equal(res, np.array([[6], [4], [0]]))


def test_dense_bp(dense_layer):
    dense_layer.forward_pass(np.array([[4.0], [2.0]]))

    feedback = np.array([[1.0], [1.0], [1.0]])
    original_weights = dense_layer.W.copy()
    exp_correction = np.array([[0.4, 0.2], [0.4, 0.2], [0.0, 0.0]])
    exp_error = np.array([[2.0], [1.0]])

    res = dense_layer.backward_pass(feedback, 0.1)

    assert_array_almost_equal(dense_layer.W, original_weights - exp_correction)
    assert_array_almost_equal(res, exp_error)


def test_flatten_2d():
    flat = Flatten(input_shape=(2, 3))

    input = np.array([[1, 2, 3], [4, 5, 6]])
    exp = np.array([[1], [2], [3], [4], [5], [6]])

    assert_array_equal(flat.forward_pass(input), exp)
    assert_array_equal(flat.backward_pass(exp, 0.01), input)


def test_flatten_3d():
    flat = Flatten(input_shape=(2, 2, 3))

    input = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])
    exp = np.array([[1], [2], [3], [4], [5], [6], [7], [8], [9], [10], [11], [12]])

    assert_array_equal(flat.forward_pass(input), exp)
    assert_array_equal(flat.backward_pass(exp, 0.01), input)


def test_dropout():
    d = Dropout(input_shape=10, factor=0.25)
    res = d.forward_pass(np.ones((10, 1)))

    assert d.mask.shape == (10, 1)
    assert sum(d.mask)[0] == 8

    assert sum(res)[0] == 8 * (1 / (1 - 0.25))

    res2 = d.backward_pass(np.ones((10, 1)), lr=0.01)

    assert sum(res2)[0] == 8 * (1 - 0.25)


def test_conv_shapes():
    c2 = Conv2D(n_kernels=6, kernel_shape=(3, 3), input_shape=(28, 28, 1))

    assert c2.output_shape == (26, 26, 6)
    assert c2.W.shape == (3, 3, 1, 6)


def test_conv_math(conv_layer):
    x = np.array([[[1], [2], [3]], [[4], [5], [6]], [[7], [8], [-20]]])
    exp = np.array([[[12, 24], [16, 32]], [[24, 48], [0, 0]]])
    res = conv_layer.forward_pass(x)

    assert res.shape == conv_layer.output_shape
    assert_array_equal(res, exp)

    error = np.ones((2, 2, 2))
    exp_W = np.array([[[[-0.1, 0.9]], [[-0.4, 0.6]]], [[[-1, 0]], [[1.6, 2.6]]]])

    # Not sure about this one
    exp_error = np.array([[[3], [6], [3]], [[6], [9], [3]], [[3], [3], [0]]])

    res2 = conv_layer.backward_pass(error, lr=0.1)

    assert res2.shape == conv_layer.input_shape
    assert_array_equal(res2, exp_error)
    print(res2, exp_error)

    assert_array_almost_equal(conv_layer.W, exp_W)

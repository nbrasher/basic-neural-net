from numpy.testing import assert_array_almost_equal
import numpy as np
import pytest

from nn.layers import Dense


@pytest.fixture
def dense_layer():
    layer = Dense(input_shape=2, output_shape=3, activation="relu")

    # Replace weights for test purposes
    layer.W = np.array([[1., 1.], [1., 0.], [-1., -1.]])
    yield layer

def test_dense(dense_layer):
    assert dense_layer.W.shape == (3, 2)
    assert dense_layer.A.shape == (2, 1)
    assert dense_layer.Z.shape == (3, 1)

def test_error():
    # Raises error on incorrect activation fn
    with pytest.raises(ValueError):
        d = Dense(input_shape=2, output_shape=4, activation="foo")

def test_dense_fp(dense_layer):
    input = np.array([[4.], [2.]])
    res = dense_layer.forward_pass(input)
    assert_array_almost_equal(dense_layer.A, input)
    assert_array_almost_equal(dense_layer.Z, np.array([[6], [4], [-6]]))
    assert_array_almost_equal(res, np.array([[6], [4], [0]]))

def test_dense_bp(dense_layer):
    dense_layer.forward_pass(np.array([[4.], [2.]]))

    feedback = np.array([[1.], [1.], [1.]])
    original_weights = dense_layer.W.copy()
    exp_correction = np.array([[.4, .2], [.4, .2], [0., 0.]])
    exp_error = np.array([[2.], [1.]])

    res = dense_layer.backward_pass(feedback, .1)

    assert_array_almost_equal(dense_layer.W, original_weights - exp_correction)
    assert_array_almost_equal(res, exp_error)

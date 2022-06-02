from math import exp

import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal

from nn.nn import SimpleNet


@pytest.fixture
def sn():
    yield SimpleNet([10, 8, 6, 4])


def test_init(sn):
    assert sn.W[0].shape == (8, 10)
    assert sn.W[1].shape == (6, 8)
    assert sn.W[2].shape == (4, 6)


def test_sigmoid(sn):
    a = np.array(range(5))
    ans = np.array([(1.0 / (1.0 + exp(-i))) for i in range(5)])

    assert_array_almost_equal(sn.sigmoid(a), ans)


def test_sigmoid_deriv(sn):
    a = np.array(range(5))
    ans = np.array([((exp(-i)) / ((1.0 + exp(-i)) ** 2)) for i in range(5)])

    assert_array_almost_equal(sn.sigmoid_deriv(a), ans)


def test_softmax(sn):
    a = np.array(range(5))
    ans = np.array([exp(i) for i in range(5)])
    ans /= sum(ans)

    assert_array_almost_equal(sn.softmax(a), ans)


def test_fp(sn):
    x = np.ones((10, 3))
    res = sn.forward_pass(x)
    assert res.shape == (4, 3)


def test_bp(sn):
    x = np.ones(10)
    y = np.random.randn(4)
    res = sn.forward_pass(x)
    delta = sn.backward_pass(y, res)

    assert len(delta) == len(sn.W)
    assert all(d.shape == w.shape for d, w in zip(delta, sn.W))

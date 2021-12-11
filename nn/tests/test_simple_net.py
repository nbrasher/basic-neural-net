from math import exp

import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal

from nn.nn import SimpleNet


@pytest.fixture
def sn():
    yield SimpleNet()


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


def test_softmax_deriv(sn):
    a = np.array(range(5))
    ans = np.array([exp(i) for i in range(5)])
    ans = (ans * (sum(ans) - ans)) / (sum(ans) ** 2)

    assert_array_almost_equal(sn.softmax_deriv(a), ans)

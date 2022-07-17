from math import log

import pytest
import numpy as np

from nn.models import MultiClassDense


@pytest.fixture
def mcd():
    yield MultiClassDense(input_shape = 10, sizes=[8, 6, 4])


def test_init(mcd):
    assert mcd.layers[0].W.shape == (8, 10)
    assert mcd.layers[1].W.shape == (6, 8)
    assert mcd.layers[2].W.shape == (4, 6)


def test_fp(mcd):
    x = np.ones((10, 3))
    res = mcd.forward_pass(x)
    assert res.shape == (4, 3)


def test_bp(mcd):
    x = np.ones(10)
    y = np.random.randn(4)
    res = mcd.forward_pass(x)
    mcd.backward_pass(error=(res - y), lr=.01)

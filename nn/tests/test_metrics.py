from math import log

import numpy as np

from nn.metrics import accuracy, cross_entropy_loss


def test_accuracy():
    yp1 = np.array([[0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.1, 0.1, 0.8]])
    yp2 = np.array([[0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.1, 0.8, 0.1]])
    yt = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

    assert accuracy(yp1, yt) == 1.0
    assert accuracy(yp2, yt) == 2.0 / 3.0


def test_cat_loss():
    yp1 = np.array([[0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.1, 0.1, 0.8]])
    yp2 = np.array([[0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.1, 0.8, 0.1]])
    yt = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

    assert cross_entropy_loss(yp1, yt) == -3 * log(0.8)
    assert cross_entropy_loss(yp2, yt) == -2 * log(0.8) - log(0.1)

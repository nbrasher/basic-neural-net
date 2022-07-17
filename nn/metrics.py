import numpy as np


def accuracy(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    predictions = [np.argmax(yp) == np.argmax(yt) for yp, yt in zip(y_pred, y_true)]

    return np.mean(predictions)


def cross_entropy_loss(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    loss = [-np.dot(np.log(yp), yt) for yp, yt in zip(y_pred, y_true)]

    return np.sum(loss)

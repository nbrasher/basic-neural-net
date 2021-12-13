# Basic Neural Network
Simple neural net using stochastic gradient descent. Coded from scratch in Numpy. This is purely an excercise in understanding he building blocks. If you cant explain it, you dont understand it.

## Usage
Useful as a learning tool.

Creates a dense neural network with sigmoid units, last layer is a softmax unit.

```python
sn = SimpleNet(sizes=[784, 128, 64, 10])
sn.train(X_train, y_train, X_test, y_test)
```

## Development
- Install requirements with `pip install -r requirements.txt` and `pip install -r requirements-dev.txt`
- Run tests with `pytest`

# Basic Neural Network
Simple neural net using stochastic gradient descent. Coded from scratch in Numpy. This is purely an excercise in understanding the building blocks.

## Usage
Useful as a learning tool.

Creates a dense neural network with sigmoid units, last layer is a softmax unit.

```python
m = MultiClassDense(input_shape=784, sizes=[128, 64, 10])
m.train(X_train, y_train, X_test, y_test)
```

## Development
- Install requirements with `pip install -r requirements.txt` and `pip install -r requirements-dev.txt`
- Run tests with `pytest`

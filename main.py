import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer

from nn.nn import SimpleNet


def main():
    try:
        X = pd.read_parquet("./data/X.parquet")
        y = pd.read_parquet("./data/y.parquet")
    except FileNotFoundError:
        X, y = fetch_openml("mnist_784", version=1, return_X_y=True)

    X = X.to_numpy() / 255.0
    y = pd.DataFrame(y)

    # Convert string labels to length-10 integer arrays
    lb = LabelBinarizer()
    lb.fit([str(i) for i in range(10)])
    y = lb.transform(y["class"])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=154
    )

    sn = SimpleNet(sizes=[784, 128, 64, 10])
    sn.train(X_train, y_train, X_test, y_test)


if __name__ == "__main__":
    main()

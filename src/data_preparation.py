import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.preprocessing import StandardScaler

def load_data():
    X = np.array([1, 2, 3, 4, 5]).reshape(-1, 1)
    y = np.array([52, 55, 65, 70, 78]).reshape(-1, 1)

    return X, y


def add_bias_column(X):
    return np.hstack([np.ones((X.shape[0], 1)), X])

def load_diabetes_data():
    diabetes = load_diabetes()

    X = diabetes.data
    y = diabetes.target.reshape(-1, 1)

    return X, y
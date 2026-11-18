import matplotlib.pyplot as plt


def plot_regression(X, y, predictions):
    plt.scatter(X[:, 1], y, color="green", label="Training data")
    plt.plot(X[:, 1], predictions, color="red", label="Regression line")

    plt.xlabel("X")
    plt.ylabel("y")
    plt.title("Linear Regression using Gradient Descent")
    plt.legend()
    plt.grid()
    plt.show()